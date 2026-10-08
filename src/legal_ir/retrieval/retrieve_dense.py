"""Truy hồi dense bằng chỉ mục vector.

File này làm gì
----------------
Nạp chỉ mục vector đã dựng ở bước trước, encode câu hỏi, tìm các chunk gần nhất, ghi ra
`ranked.jsonl` đúng định dạng mà bước fusion cần.

Pipeline stage
--------------
Sau khi dựng chỉ mục dense, cùng nhóm với retriever BM25.

Input
-----
  --index-dir  thư mục chỉ mục vector
  --queries    file câu hỏi cần truy hồi
  --output-dir thư mục ghi kết quả
  --top-k      số kết quả mỗi câu
  --chunk-depth độ sâu chunk theo từng encoder

Output
-------
  <output-dir>/ranked.jsonl    mỗi dòng: query_id, ranked, method, thang điểm
  <output-dir>/report.json     tham số truy hồi, sha256

Model/class
-----------
  sentence-transformers — encoder theo từng nhánh: VAE, BGE-M3 hoặc F2LLM (LoRA).
  Cần CUDA cho phần encode; phần tìm kiếm vector chạy được trên CPU.

Command
-------
  python retrieve_dense.py --index-dir <R>/indexes/bge_m3 \
      --queries <W>/queries/private.jsonl --output-dir <R>/outputs/bgem3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from legal_ir.retrieval.index_dense import build_dense_encoder
from legal_ir.core.commons import (
    aggregate_chunk_hits,
    atomic_write_json,
    atomic_write_jsonl,
    build_submission,
    fill_with_fallback,
    load_chunk_table,
    load_index_manifest,
    load_queries,
    select_answer_ids,
    validate_view_weights,
)


CHUNK_DEPTH_BY_ENCODER = {
    "vae-v2": 2048,
    "bge-m3": 8192,
    "f2llm-1.7b": 8192,
}
DEFAULT_CHUNK_DEPTH = 2048


def log(message: str) -> None:
    print(message, flush=True)


def view_weights(args: argparse.Namespace) -> dict[str, float]:
    return validate_view_weights(
        {
            "packed": args.packed_weight,
            "article": args.article_weight,
            "metadata": args.metadata_weight,
            "recovery": args.recovery_weight,
            "short": args.packed_weight,
            "unknown": 1.0,
        }
    )


def move_index_to_device(faiss: Any, cpu_index: Any, args: argparse.Namespace):
    gpu_api_available = all(
        hasattr(faiss, name)
        for name in ("StandardGpuResources", "index_cpu_to_gpu", "GpuClonerOptions")
    )

    if args.device == "cpu":
        log("[FAISS] exact search device=cpu (forced)")
        return cpu_index, "cpu", None

    if not gpu_api_available:
        if args.device == "gpu":
            raise RuntimeError(
                "--device gpu requested, but this FAISS build has no GPU API. "
                "Install faiss-gpu or use --device cpu/auto."
            )
        log("[FAISS] GPU API not found; exact search will run on CPU")
        return cpu_index, "cpu", None

    try:
        resources = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.useFloat16 = bool(args.faiss_fp16)
        search_index = faiss.index_cpu_to_gpu(
            resources,
            args.gpu_id,
            cpu_index,
            options,
        )
        actual = f"gpu:{args.gpu_id}"
        log(
            f"[FAISS] exact search device={actual} "
            f"temporary_clone_fp16={bool(args.faiss_fp16)}"
        )
        return search_index, actual, resources
    except Exception as exc:
        if args.device == "gpu":
            raise RuntimeError(
                f"Failed to move FAISS index to GPU {args.gpu_id}: {exc}"
            ) from exc
        log(
            "[FAISS] GPU clone failed in auto mode; falling back to exact CPU "
            f"search ({type(exc).__name__}: {exc})"
        )
        return cpu_index, "cpu", None


def run(args: argparse.Namespace) -> int:
    if not args.queries.is_file():
        raise FileNotFoundError(f"Query file not found: {args.queries}")
    if not args.index_dir.is_dir():
        raise FileNotFoundError(f"Index directory not found: {args.index_dir}")
    if not 1 <= args.top_k <= 5:
        raise ValueError("The official evaluator allows at most five answers")
    if args.context_pool < args.top_k:
        raise ValueError("context-pool must be >= top-k")
    if args.chunk_depth is not None and args.chunk_depth < args.context_pool:
        raise ValueError("chunk-depth must be >= context-pool")
    if args.query_batch_size <= 0:
        raise ValueError("query-batch-size must be positive")
    if args.encode_batch_size <= 0:
        raise ValueError("encode-batch-size must be positive")
    if args.threads <= 0:
        raise ValueError("threads must be positive")
    if not 0.0 <= args.score_ratio_cutoff <= 1.0:
        raise ValueError("score-ratio-cutoff must be between 0 and 1")

    weights = view_weights(args)

    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("FAISS and NumPy are required for retrieval") from exc

    log(
        "RETRIEVE_NMODELS_START "
        f"queries={args.queries} index_dir={args.index_dir} "
        f"faiss_device={args.device} encoder_device={args.encoder_device}"
    )


    manifest = load_index_manifest(args.index_dir, backend="faiss")
    embedding = manifest.get("embedding", {})

    indexed_model = str(embedding.get("model", "")).strip()
    model = str(args.model or indexed_model).strip()
    if args.model and model != indexed_model and not args.allow_model_mismatch:
        raise RuntimeError(
            f"Model mismatch: index={indexed_model!r}, query encoder={model!r}. "
            "Omit --model to reuse the index manifest automatically, or use "
            "--allow-model-mismatch only for a verified identical checkpoint."
        )

    indexed_dims = int(embedding.get("dims", -1))
    if args.dims is not None and args.dims != indexed_dims:
        raise RuntimeError(
            f"Dimension mismatch: index={indexed_dims}, requested={args.dims}"
        )

    indexed_max_length = int(embedding.get("max_length", 0))
    max_length = int(args.max_length or indexed_max_length)

    if indexed_dims <= 0 or max_length <= 0 or not model:
        raise ValueError("Dense index manifest lacks a valid embedding contract")

    encoder_id = str(embedding.get("encoder_id", "")).strip()
    if not encoder_id:
        raise ValueError("Index manifest missing encoder_id in embedding")

    log(
        "[CONTRACT] "
        f"encoder_id={encoder_id} model={model} dims={indexed_dims} max_length={max_length} "
        f"index_chunks={manifest.get('chunks')}"
    )


    index_file = args.index_dir / manifest["faiss"]["index_file"]
    if not index_file.is_file():
        raise FileNotFoundError(f"FAISS index file not found: {index_file}")

    log(f"[FAISS] loading persisted index: {index_file}")
    cpu_index = faiss.read_index(str(index_file))
    table = load_chunk_table(args.index_dir, mmap=args.mmap_sidecars)

    manifest_chunks = int(manifest["chunks"])
    if int(cpu_index.ntotal) != len(table) or int(cpu_index.ntotal) != manifest_chunks:
        raise RuntimeError(
            "FAISS index/sidecar count mismatch: "
            f"faiss={cpu_index.ntotal}, sidecars={len(table)}, "
            f"manifest={manifest_chunks}"
        )
    if int(cpu_index.d) != indexed_dims or not cpu_index.is_trained:
        raise RuntimeError("FAISS index does not satisfy its embedding manifest")
    if (
        hasattr(faiss, "METRIC_INNER_PRODUCT")
        and cpu_index.metric_type != faiss.METRIC_INNER_PRODUCT
    ):
        raise RuntimeError(
            "Dense FAISS index must use inner product over normalized vectors"
        )
    if hasattr(faiss, "omp_set_num_threads"):
        faiss.omp_set_num_threads(args.threads)

    search_index, actual_faiss_device, _gpu_resources = move_index_to_device(
        faiss,
        cpu_index,
        args,
    )


    log(f"[ENCODER] constructing query encoder on {args.encoder_device}")
    encoder = build_dense_encoder(
        encoder_id=encoder_id,
        model_name=model,
        dims=indexed_dims,
        max_length=max_length,
        batch_size=args.encode_batch_size,
        fp16=args.fp16,
        device=args.encoder_device,
    )

    queries = load_queries(args.queries)
    if not queries:
        raise ValueError(f"Query file is empty: {args.queries}")

    chunk_depth = (
        args.chunk_depth
        if args.chunk_depth is not None
        else CHUNK_DEPTH_BY_ENCODER.get(encoder_id, DEFAULT_CHUNK_DEPTH)
    )
    if chunk_depth < args.context_pool:
        raise ValueError("chunk-depth must be >= context-pool")
    if args.max_chunk_depth < chunk_depth:
        raise ValueError("max-chunk-depth must be >= chunk-depth")

    indexed_rows = int(cpu_index.ntotal)
    initial_depth = min(chunk_depth, indexed_rows)
    maximum_depth = min(args.max_chunk_depth, indexed_rows)
    rankings: dict[str, list[dict[str, Any]]] = {}
    effective_depths: dict[str, int] = {}
    started = time.monotonic()
    warned_gpu_fallback = False

    def search(vectors: Any, depth: int):
        nonlocal warned_gpu_fallback
        try:
            return search_index.search(vectors, depth)
        except RuntimeError:


            if actual_faiss_device == "cpu":
                raise
            if not warned_gpu_fallback:
                log(
                    "[FAISS] GPU search rejected requested depth; using exact CPU "
                    "IndexFlatIP for large adaptive searches"
                )
                warned_gpu_fallback = True
            return cpu_index.search(vectors, depth)

    for start in range(0, len(queries), args.query_batch_size):
        batch = queries[start : start + args.query_batch_size]

        vectors = encoder.encode([query["question"] for query in batch], query=True)
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)

        scores_batch, indices_batch = search(vectors, initial_depth)

        for position, query in enumerate(batch):
            depth = initial_depth
            indices = indices_batch[position]
            scores = scores_batch[position]

            hits = aggregate_chunk_hits(
                indices,
                scores,
                table,
                context_pool=args.context_pool,
                view_weights=weights,
                skip_nonpositive=False,
            )

            while len(hits) < args.top_k and depth < maximum_depth:
                depth = min(depth * 2, maximum_depth)
                scores_one, indices_one = search(
                    vectors[position : position + 1],
                    depth,
                )
                hits = aggregate_chunk_hits(
                    indices_one[0],
                    scores_one[0],
                    table,
                    context_pool=args.context_pool,
                    view_weights=weights,
                    skip_nonpositive=False,
                )

            fill_with_fallback(hits, table.contexts, args.top_k)
            rankings[query["query_id"]] = hits
            effective_depths[query["query_id"]] = depth

        log(
            f"[PROGRESS] encoded/retrieved "
            f"{min(start + len(batch), len(queries))}/{len(queries)} queries "
            f"encode_batch={encoder.batch_size}"
        )

    answers = {
        query["query_id"]: select_answer_ids(
            rankings[query["query_id"]],
            top_k=args.top_k,
            score_ratio_cutoff=args.score_ratio_cutoff,
        )
        for query in queries
    }

    query_order = [query["query_id"] for query in queries]
    submission = build_submission(query_order, answers)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.json"
    ranked_path = args.output_dir / "ranked.jsonl"
    metrics_path = args.output_dir / "metrics.json"

    atomic_write_json(submission_path, submission)
    atomic_write_jsonl(
        ranked_path,
        (
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "method": f"dense-faiss:{model}",
                "ranked": [
                    hit["context_id"] for hit in rankings[query["query_id"]]
                ],
                "hits": (
                    rankings[query["query_id"]]
                    if not args.debug_top
                    else rankings[query["query_id"]][: args.debug_top]
                ),
            }
            for query in queries
        ),
    )

    elapsed = time.monotonic() - started
    report: dict[str, Any] = {
        "method": f"dense-faiss-exact:{model}",
        "index_dir": str(args.index_dir.resolve()),
        "queries": str(args.queries.resolve()),
        "model": model,
        "model_source": "cli" if args.model else "index_manifest",
        "dims": indexed_dims,
        "max_length": max_length,
        "encoder_device": args.encoder_device,
        "encoder_fp16": bool(encoder.fp16),
        "encoder_prompt_verified": bool(encoder._prompt_verified),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "context_pool": args.context_pool,
        "chunk_depth": chunk_depth,
        "max_chunk_depth": args.max_chunk_depth,
        "maximum_effective_chunk_depth": max(effective_depths.values()),
        "score_ratio_cutoff": args.score_ratio_cutoff,
        "view_weights": weights,
        "faiss_device": actual_faiss_device,
        "faiss_fp16_storage": args.faiss_fp16,
        "persisted_index_exact": True,
        "search_exact_over_stored_float32": not args.faiss_fp16,
        "elapsed_seconds": elapsed,
        "submission": str(submission_path.resolve()),
        "ranked": str(ranked_path.resolve()),
    }

    atomic_write_json(metrics_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    log(f"RETRIEVE_NMODELS_DONE rc=0 elapsed={elapsed:.2f}s")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--dims", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--allow-model-mismatch", action="store_true")

    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument(
        "--encoder-device",
        default="cuda:0",
        help="Dense query encoder device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="FAISS search device; independent from --encoder-device.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--faiss-fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Use fp16 storage only for the temporary FAISS GPU clone."
        ),
    )
    parser.add_argument(
        "--mmap-sidecars",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-pool", type=int, default=200)
    parser.add_argument(
        "--chunk-depth",
        type=int,
        help="Override the per-encoder depth default (vae 2048, bge-m3/f2llm 8192)",
    )
    parser.add_argument("--max-chunk-depth", type=int, default=32768)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--score-ratio-cutoff", type=float, default=0.0)

    parser.add_argument("--packed-weight", type=float, default=1.0)
    parser.add_argument("--article-weight", type=float, default=1.0)
    parser.add_argument("--metadata-weight", type=float, default=1.0)
    parser.add_argument("--recovery-weight", type=float, default=1.0)
    parser.add_argument(
        "--debug-top",
        type=int,
        default=0,
        help="ghi đủ pool khi 0 (chunk_id đủ cho rerank); >0 chỉ ghi top-N",
    )
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except KeyboardInterrupt:
        print("ERROR: interrupted by user/scheduler", file=sys.stderr, flush=True)
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
