"""Dựng chỉ mục vector trên corpus chunk.

File này làm gì
----------------
Encode toàn bộ chunk trong corpus bằng một encoder, lưu thành chỉ mục vector để bước truy
hồi dense dùng. Mỗi encoder có một tham số `--encoder` khác nhau: `vietnamese-embedding`
(VAE), `bge-m3`, `f2llm-1.7b`.

Pipeline stage
--------------
Sau bước dựng corpus, song song với việc dựng chỉ mục BM25.

Input
-----
  --corpus     retrieval_chunks.jsonl
  --index-dir  thư mục ghi chỉ mục
  --encoder    tên encoder: vietnamese-embedding | bge-m3 | f2llm-1.7b
  --model-name checkpoint tinh chỉnh (bỏ trống với VAE vì dùng model gốc)

Output
-------
  <index-dir>/   chỉ mục vector và báo cáo tham số đã dùng

Model/class
-----------
  sentence-transformers — VAE (`AITeamVN/Vietnamese_Embedding_v2`), BGE-M3 (`BAAI/bge-m3`
  đã tinh chỉnh), hoặc F2LLM (`codefuse-ai/F2LLM-1.7B` cộng adapter LoRA đã tinh chỉnh).
  Cần CUDA để encode; cần môi trường riêng cho F2LLM.

Command
-------
  python index_dense.py --encoder bge-m3 --model-name checkpoints/bge_finetune_epoch1 \
      --corpus <W>/corpus/retrieval_chunks.jsonl --index-dir <R>/indexes/bge_m3
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Sequence

from legal_ir.core.commons import (
    INDEX_FORMAT_VERSION,
    ChunkTableBuilder,
    atomic_write_json,
    commit_build_directory,
    create_build_directory,
    discard_build_directory,
    expected_retrieval_count,
    index_file_inventory,
    iter_retrieval_rows,
    l2_normalize,
    sha256_file,
)


def log(message: str) -> None:
    print(message, flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


class BgeM3Model:
    MODEL_NAME = "BAAI/bge-m3"
    ENCODER_ID = "bge-m3"

    def __init__(
        self,
        model: str = MODEL_NAME,
        max_length: int = 2048,
        batch_size: int = 128,
        dims: int = 1024,
        fp16: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.model_name = str(model)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.dims = int(dims)
        self.device = str(device)
        self.encoder_backend = "FlagEmbedding"
        self.encoder_version = package_version("FlagEmbedding")

        log("[BGE] importing torch ...")
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed in the active environment.") from exc

        self._torch = torch
        log(f"[BGE] torch={torch.__version__}")

        if self.device.startswith("cuda"):
            log("[BGE] checking CUDA ...")
            cuda_available = bool(torch.cuda.is_available())
            log(
                "[BGE] "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
                f"cuda_available={cuda_available} "
                f"cuda_device_count={torch.cuda.device_count()}"
            )
            if not cuda_available:
                raise RuntimeError(
                    f"device={self.device} was requested, but torch.cuda.is_available() is False. "
                    "Refusing to fall back to CPU for a full-corpus build."
                )

            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception as exc:
                raise RuntimeError("CUDA is visible but cuda:0 could not be queried.") from exc
            log(f"[BGE] logical cuda:0 -> {gpu_name}")
        elif self.device != "cpu":
            raise ValueError(f"Unsupported device: {self.device!r}; use 'cuda:0' or 'cpu'.")

        self.fp16 = bool(fp16 and self.device.startswith("cuda"))
        if fp16 and not self.fp16:
            log("[BGE] WARNING: fp16 requested on non-CUDA device; using float32.")

        log("[BGE] importing FlagEmbedding.BGEM3FlagModel ...")
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError as exc:
            raise RuntimeError(
                "FlagEmbedding is not installed or failed to import. "
                "Install FlagEmbedding==1.4.0 in the active environment."
            ) from exc
        log(f"[BGE] FlagEmbedding={package_version('FlagEmbedding')} imported")

        log(
            "[BGE] loading model "
            f"name={self.model_name} device={self.device} "
            f"max_length={self.max_length} batch_size={self.batch_size} "
            f"fp16={self.fp16}"
        )


        self.model = BGEM3FlagModel(
            self.model_name,
            use_fp16=self.fp16,
            normalize_embeddings=True,
            devices=self.device,
            batch_size=self.batch_size,
            passage_max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )


        self._prompt_verified = True

        log(
            "[BGE] model loaded successfully "
            f"name={self.model_name} device={self.device} fp16={self.fp16}"
        )

    def encode(self, texts: Sequence[str], *, query: bool = False):
        if not texts:
            raise ValueError("Cannot encode an empty text batch.")

        log(
            f"[BGE] encode start outer_chunks={len(texts):,} "
            f"inner_batch={self.batch_size}"
        )
        started = time.monotonic()

        try:
            result = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )
        except self._torch.cuda.OutOfMemoryError as exc:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
            raise RuntimeError(
                "CUDA OOM while encoding. Re-submit with a smaller "
                "ENCODE_BATCH_SIZE, for example 8 or 16."
            ) from exc

        if not isinstance(result, dict) or result.get("dense_vecs") is None:
            raise RuntimeError("BGE-M3 did not return result['dense_vecs'].")


        vectors = l2_normalize(result["dense_vecs"])

        if vectors.shape != (len(texts), self.dims):
            raise ValueError(
                f"BGE-M3 returned shape {vectors.shape}; "
                f"expected ({len(texts)}, {self.dims})."
            )

        elapsed = time.monotonic() - started
        rate = len(texts) / max(elapsed, 1e-9)
        log(
            f"[BGE] encode done outer_chunks={len(texts):,} "
            f"elapsed={elapsed:.2f}s rate={rate:.2f} chunks/s"
        )
        return vectors


def _load_sentence_transformer(model_name: str, max_length: int, fp16: bool, device: str):
    log("[ST] importing sentence_transformers ...")
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for this encoder. "
            "Install it in the active environment."
        ) from exc

    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"Encoder device {device!r} requested but CUDA is unavailable; "
            "refusing CPU fallback"
        )

    effective_fp16 = bool(fp16 and str(device).startswith("cuda"))
    dtype = None
    if effective_fp16:
        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )
    model_kwargs = {}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype

    log(
        "[ST] loading model "
        f"name={model_name} device={device} dtype={dtype} max_length={max_length}"
    )
    model = SentenceTransformer(model_name, device=device, model_kwargs=model_kwargs)
    model.max_seq_length = int(max_length)
    return torch, model, effective_fp16


class F2LlmModel:
    ENCODER_ID = "f2llm-1.7b"

    def __init__(
        self,
        model: str,
        max_length: int = 512,
        batch_size: int = 128,
        dims: int = 2048,
        fp16: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.dims = int(dims)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.encoder_backend = "sentence-transformers"
        self.encoder_version = package_version("sentence_transformers")
        self.fp16 = bool(fp16)


        import json
        from pathlib import Path
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer

        checkpoint_dir = Path(model)
        adapter_config_path = checkpoint_dir / "adapter_config.json"
        if not adapter_config_path.exists():
            raise FileNotFoundError(
                f"{checkpoint_dir} khong co adapter_config.json - khong the load LoRA adapter. "
                f"Can base_model_name_or_path de load base model."
            )
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_name = adapter_config.get("base_model_name_or_path")
        if not base_name:
            raise ValueError(f"{adapter_config_path} thieu key 'base_model_name_or_path'")

        log(f"[F2LLM] loading base model {base_name} + LoRA adapter from {checkpoint_dir}")


        st_model = SentenceTransformer(base_name, device=device)
        st_model.max_seq_length = int(max_length)


        transformer_module = st_model[0]
        transformer_module.model = PeftModel.from_pretrained(
            transformer_module.model, str(checkpoint_dir)
        ).merge_and_unload()


        F2LLM_QUERY_PROMPT = (
            "Instruct: Given a question, retrieve passages that can help answer the question.\n"
            "Query: "
        )
        st_model.prompts["query"] = F2LLM_QUERY_PROMPT
        st_model.prompts.setdefault("document", "")

        self.model = st_model
        self.model_name = str(checkpoint_dir)
        self._torch = __import__("torch")
        self.fp16 = fp16
        self._prompt_verified = True

        log(
            f"[F2LLM] model loaded successfully "
            f"backend={self.encoder_backend} version={self.encoder_version}"
        )

    def encode(self, texts: Sequence[str], *, query: bool = False):
        if not texts:
            raise ValueError("Cannot encode an empty text batch.")

        method = self.model.encode_query if query else self.model.encode_document
        log(
            f"[F2LLM] encode start outer_chunks={len(texts):,} query={query} "
            f"inner_batch={self.batch_size} method={method.__name__}"
        )
        started = time.monotonic()

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for dense embeddings") from exc

        vectors = method(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))

        if vectors.shape != (len(texts), self.dims):
            raise ValueError(
                f"F2LLM returned shape {vectors.shape}; expected ({len(texts)}, {self.dims})."
            )

        elapsed = time.monotonic() - started
        rate = len(texts) / max(elapsed, 1e-9)
        log(
            f"[F2LLM] encode done outer_chunks={len(texts):,} "
            f"elapsed={elapsed:.2f}s rate={rate:.2f} chunks/s"
        )
        return vectors


class VaeModel:

    MODEL_NAME = "AITeamVN/Vietnamese_Embedding_v2"
    ENCODER_ID = "vae-v2"

    def __init__(
        self,
        model: str = MODEL_NAME,
        max_length: int = 2048,
        batch_size: int = 16,
        dims: int = 1024,
        fp16: bool = True,
        device: str = "cuda:0",
    ) -> None:
        self.model_name = str(model)
        self.dims = int(dims)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.device = str(device)
        self.encoder_backend = "sentence-transformers"
        self.encoder_version = package_version("sentence_transformers")

        self._torch, self.model, self.fp16 = _load_sentence_transformer(
            self.model_name, self.max_length, fp16, self.device
        )


        self._prompt_verified = True

        log(
            f"[VAE] model loaded successfully "
            f"backend={self.encoder_backend} version={self.encoder_version}"
        )

    def encode(self, texts: Sequence[str], *, query: bool = False):
        if not texts:
            raise ValueError("Cannot encode an empty text batch.")

        log(
            f"[VAE] encode start outer_chunks={len(texts):,} query={query} "
            f"inner_batch={self.batch_size} method=encode"
        )
        started = time.monotonic()

        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required for dense embeddings") from exc

        vectors = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))

        if vectors.shape != (len(texts), self.dims):
            raise ValueError(
                f"VAE returned shape {vectors.shape}; expected ({len(texts)}, {self.dims})."
            )

        elapsed = time.monotonic() - started
        rate = len(texts) / max(elapsed, 1e-9)
        log(
            f"[VAE] encode done outer_chunks={len(texts):,} "
            f"elapsed={elapsed:.2f}s rate={rate:.2f} chunks/s"
        )
        return vectors


ENCODER_REGISTRY: dict[str, type] = {
    "bge-m3": BgeM3Model,
    "f2llm-1.7b": F2LlmModel,
    "vae-v2": VaeModel,
}


def build_dense_encoder(
    encoder_id: str,
    batch_size: int,
    fp16: bool,
    device: str,
    model_name: str | None = None,
    max_length: int | None = None,
    dims: int | None = None,
):
    try:
        encoder_cls = ENCODER_REGISTRY[encoder_id]
    except KeyError as exc:
        raise ValueError(
            f"Unknown encoder_id={encoder_id!r}. "
            f"Available encoders: {sorted(ENCODER_REGISTRY)}"
        ) from exc

    kwargs = {
        "fp16": fp16,
        "device": device,
    }
    if batch_size is not None:
        kwargs["batch_size"] = batch_size
    if model_name is not None:
        kwargs["model"] = model_name
    if max_length is not None:
        kwargs["max_length"] = max_length
    if dims is not None:
        kwargs["dims"] = dims

    return encoder_cls(**kwargs)


def run(args: argparse.Namespace) -> int:
    log(
        "INDEX_NMODELS_START "
        f"corpus={args.corpus} index_dir={args.index_dir} "
        f"encoder={args.encoder} device={args.device}"
    )

    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)

    if args.encode_batch_size is not None and args.encode_batch_size <= 0:
        raise ValueError("encode-batch-size must be positive")
    if args.encode_buffer_chunks <= 0:
        raise ValueError("encode-buffer-chunks must be positive")

    log("[INIT] importing FAISS and NumPy ...")
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "FAISS and NumPy are required. Install faiss-cpu==1.12.0 "
            "and numpy in the active environment."
        ) from exc

    log(
        f"[INIT] faiss={getattr(faiss, '__version__', 'unknown')} "
        f"numpy={np.__version__}"
    )

    expected = args.expected_count
    if expected is None:
        log("[INIT] resolving expected retrieval-row count ...")
        expected = expected_retrieval_count(args.corpus)
    log(f"[INIT] expected retrieval rows={expected:,}")


    log("[INIT] constructing dense encoder ...")
    encoder = build_dense_encoder(
        encoder_id=args.encoder,
        batch_size=args.encode_batch_size,
        fp16=args.fp16,
        device=args.device,
        model_name=getattr(args, "model_name", None),
        max_length=getattr(args, "max_length", None),
        dims=getattr(args, "dims", None),
    )
    log("[INIT] dense encoder ready")

    log(f"[INIT] creating build directory for {args.index_dir} ...")
    build_dir: Path | None = None

    try:
        build_dir = create_build_directory(args.index_dir, args.recreate)
        log(f"[INIT] build_dir={build_dir}")

        log("[INIT] creating chunk sidecar table ...")
        table = ChunkTableBuilder()
        log("[INIT] chunk sidecar table ready")

        log(
            f"[INIT] creating exact FAISS IndexFlatIP(d={encoder.dims}) ..."
        )
        index = faiss.IndexFlatIP(encoder.dims)
        if index.d != encoder.dims or not index.is_trained:
            raise RuntimeError("Unexpected initial FAISS IndexFlatIP state.")
        log("[INIT] FAISS index ready")

        started = time.monotonic()
        first_vector = None
        rows: list[dict[str, str]] = []

        def flush_rows() -> None:
            nonlocal first_vector

            if not rows:
                return

            batch_n = len(rows)
            before = int(index.ntotal)
            log(
                f"[INDEX] flushing chunks={batch_n:,} "
                f"indexed_before={before:,}"
            )

            vectors = encoder.encode([row["text"] for row in rows], query=False)
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)

            if first_vector is None:
                first_vector = vectors[0].copy()

            index.add(vectors)

            for row in rows:
                table.add(row["chunk_id"], row["context_id"])

            rows.clear()

            elapsed = time.monotonic() - started
            total = int(index.ntotal)
            overall_rate = total / max(elapsed, 1e-9)
            progress = (
                f"{100.0 * total / expected:.2f}%"
                if expected
                else "unknown"
            )
            log(
                f"[INDEX] indexed={total:,}/{expected:,} "
                f"progress={progress} "
                f"overall_rate={overall_rate:.2f} chunks/s "
                f"adaptive_batch={encoder.batch_size}"
            )

        log("[INDEX] reading corpus and encoding chunks ...")
        for row in iter_retrieval_rows(args.corpus):
            rows.append(row)
            if len(rows) >= args.encode_buffer_chunks:
                flush_rows()

        flush_rows()

        row_count = int(index.ntotal)
        if row_count == 0 or first_vector is None:
            raise ValueError("Corpus contains no retrieval chunks.")

        if row_count != len(table):
            raise AssertionError(
                f"FAISS/chunk sidecar alignment failed: "
                f"index={row_count}, sidecar={len(table)}."
            )

        if expected is not None and row_count != expected:
            raise ValueError(
                f"Corpus rows={row_count:,}, preprocessing manifest "
                f"expected={expected:,}."
            )

        log("[VERIFY] running exact self-retrieval smoke test ...")
        distances, indices = index.search(first_vector.reshape(1, -1), 1)
        smoke_row = int(indices[0, 0])
        smoke_score = float(distances[0, 0])

        if not 0 <= smoke_row < row_count:
            raise RuntimeError(
                f"FAISS smoke retrieval returned invalid row={smoke_row}."
            )
        if not np.isfinite(smoke_score):
            raise RuntimeError(
                f"FAISS smoke retrieval returned non-finite score={smoke_score}."
            )


        if smoke_row != 0:
            raise RuntimeError(
                f"FAISS smoke self-retrieval returned row={smoke_row}, expected 0 "
                "(searching the first indexed vector should retrieve itself)."
            )
        if smoke_score < 0.99:
            raise RuntimeError(
                f"FAISS smoke self-retrieval score={smoke_score:.6f} < 0.99 — "
                "vectors likely not correctly normalized before index.add()."
            )

        log(
            f"[VERIFY] smoke_row={smoke_row} "
            f"smoke_score={smoke_score:.6f}"
        )

        index_path = build_dir / "index.faiss"
        log(f"[WRITE] writing FAISS index -> {index_path}")
        faiss.write_index(index, str(index_path))

        log("[WRITE] writing chunk/context sidecar ...")
        table.save(build_dir)

        elapsed = time.monotonic() - started

        torch = encoder._torch
        gpu_name = None
        if encoder.device.startswith("cuda") and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)

        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "status": "complete",
            "backend": "faiss",
            "pipeline": "legal-ir-retrieval-compact",
            "document_key": "context_id",
            "candidate_key": "chunk_id",
            "aggregation": "max_chunk_cosine_per_context",
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": None if args.skip_sha256 else sha256_file(args.corpus),
            "chunks": row_count,
            "contexts": len(table.contexts),
            "faiss": {
                "library_version": str(getattr(faiss, "__version__", "unknown")),
                "index_class": "IndexFlatIP",
                "index_file": "index.faiss",
                "metric": "inner_product",
                "exact": True,
                "storage_dtype": "float32",
                "bytes_per_vector": 4 * encoder.dims,
            },
            "embedding": {
                "encoder_id": encoder.ENCODER_ID,
                "model": encoder.model_name,
                "encoder_backend": encoder.encoder_backend,
                "encoder_version": encoder.encoder_version,
                "flagembedding_version": package_version("FlagEmbedding"),
                "dims": encoder.dims,
                "max_length": encoder.max_length,
                "normalization": "L2",
                "similarity": "cosine_via_inner_product",
                "encode_fp16": encoder.fp16,
                "encode_batch_size": encoder.batch_size,
                "device": encoder.device,
                "prompt_verified": encoder._prompt_verified,
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": str(torch.__version__),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu_name": gpu_name,
                "elapsed_seconds": elapsed,
            },
            "smoke_row": smoke_row,
            "smoke_score": smoke_score,
        }

        log("[WRITE] writing manifest ...")
        atomic_write_json(build_dir / "manifest.json", manifest)
        manifest["files"] = index_file_inventory(build_dir)
        atomic_write_json(build_dir / "manifest.json", manifest)

        log(f"[COMMIT] committing build -> {args.index_dir}")
        commit_build_directory(build_dir, args.index_dir, args.recreate)
        build_dir = None

        if args.report:
            log(f"[WRITE] writing report -> {args.report}")
            atomic_write_json(args.report, manifest)

        print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
        log("INDEX_NMODELS_DONE rc=0")
        return 0

    except Exception:
        if build_dir is not None:
            log(f"[CLEANUP] discarding incomplete build directory: {build_dir}")
            discard_build_directory(build_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument(
        "--encoder",
        choices=tuple(ENCODER_REGISTRY),
        required=True,
    )
    parser.add_argument("--model-name", type=str, help="Model name or path (overrides encoder default)")
    parser.add_argument("--max-length", type=int, help="Max sequence length (overrides encoder default)")
    parser.add_argument("--dims", type=int, help="Embedding dimensions (overrides encoder default)")
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        help="Batch size (overrides encoder default)",
    )
    parser.add_argument("--encode-buffer-chunks", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    return parser

def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        log("ERROR: interrupted by user/scheduler")
        return 130
    except Exception as exc:
        print(
            f"ERROR: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
