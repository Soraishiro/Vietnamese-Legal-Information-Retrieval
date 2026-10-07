"""Dựng chỉ mục BM25 trên corpus chunk.

File này làm gì
----------------
Đọc `retrieval_chunks.jsonl`, tách từ, dựng chỉ mục BM25 theo chuẩn Lucene để bước truy
hồi dùng. Ghi kèm tham số đã dùng để bước sau đọc lại được đúng cấu hình.

Pipeline stage
--------------
Sau bước dựng corpus, trước bước truy hồi BM25.

Input
-----
  --corpus    retrieval_chunks.jsonl
  --index-dir thư mục ghi chỉ mục
  Cùng các tham số: profile từ vựng, b, backend, cách chia đoạn

Output
-------
  <index-dir>/   chỉ mục BM25 và file tham số đã dùng

Model/class
-----------
Không dùng mô hình neural. Dùng bm25s, chuẩn Lucene với b=0.55, backend numpy.

Command
-------
  python index_bm25.py --corpus <W>/corpus/retrieval_chunks.jsonl --index-dir <R>/indexes/bm25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterator

from commons import (
    INDEX_FORMAT_VERSION,
    LEXICAL_PROFILE,
    ChunkTableBuilder,
    atomic_write_json,
    build_word_segmenter,
    commit_build_directory,
    create_build_directory,
    discard_build_directory,
    expected_retrieval_count,
    index_file_inventory,
    iter_retrieval_rows,
    lexical_document_tokens,
    lexical_profile_signature,
    lexical_tokens_to_text,
    sha256_file,
)


def whitespace_splitter(value: str) -> list[str]:
    return value.split()


def run(args: argparse.Namespace) -> int:
    if not args.corpus.is_file():
        raise FileNotFoundError(args.corpus)
    if args.k1 < 0.0 or not 0.0 <= args.b <= 1.0:
        raise ValueError("BM25 requires k1 >= 0 and 0 <= b <= 1")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")

    try:
        import bm25s
        import numpy as np
        from bm25s.tokenization import Tokenized, Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing BM25 dependencies. Install requirements.txt before indexing."
        ) from exc

    expected = args.expected_count
    if expected is None:
        expected = expected_retrieval_count(args.corpus)

    build_dir = create_build_directory(args.index_dir, args.recreate)
    started = time.monotonic()
    table = ChunkTableBuilder()
    tokenizer = Tokenizer(
        lower=False,
        splitter=whitespace_splitter,
        stopwords=[],
        stemmer=None,
    )
    segmenter = build_word_segmenter(args.word_segmenter, args.vncorenlp_dir)
    first_lexical_text = ""

    def lexical_text_stream() -> Iterator[str]:
        nonlocal first_lexical_text
        for row_number, row in enumerate(iter_retrieval_rows(args.corpus), 1):
            tokens = lexical_document_tokens(
                row["text"],
                profile=args.lexical_profile,
                segmenter=segmenter,
            )
            if not tokens:
                raise ValueError(f"Chunk has no lexical tokens: {row['chunk_id']}")
            lexical_text = lexical_tokens_to_text(tokens)
            if not first_lexical_text:
                first_lexical_text = lexical_text
            table.add(row["chunk_id"], row["context_id"])
            if row_number % args.progress_every == 0:
                print(f"Tokenized {row_number:,} chunks", flush=True)
            yield lexical_text

    try:
        token_stream = tokenizer.tokenize(
            lexical_text_stream(),
            update_vocab=True,
            return_as="stream",
            allow_empty=True,
            show_progress=False,
        )
        corpus_ids = [np.asarray(ids, dtype=np.int32) for ids in token_stream]
        row_count = len(corpus_ids)
        if row_count == 0:
            raise ValueError("Corpus contains no retrieval chunks")
        if row_count != len(table):
            raise AssertionError("BM25 token/chunk sidecar alignment failed")
        if expected is not None and row_count != expected:
            raise ValueError(
                f"Corpus rows={row_count:,}, preprocessing manifest expected={expected:,}"
            )

        vocab = tokenizer.get_vocab_dict()
        if not vocab or max(vocab.values()) >= np.iinfo(np.int32).max:
            raise ValueError("BM25 vocabulary does not fit in int32 token IDs")
        print(
            f"Building BM25S: chunks={row_count:,}, contexts={len(table.contexts):,}, "
            f"vocab={len(vocab):,}, method={args.method}, k1={args.k1}, b={args.b}",
            flush=True,
        )
        retriever = bm25s.BM25(
            k1=args.k1,
            b=args.b,
            method=args.method,
            dtype="float32",
            int_dtype="int32",
            backend=args.backend,
            csc_backend="numpy",
        )
        retriever.index(
            Tokenized(ids=corpus_ids, vocab=vocab),
            show_progress=True,
            leave_progress=False,
        )
        retriever.save(build_dir, show_progress=False)
        tokenizer.save_vocab(str(build_dir))
        table.save(build_dir)

        smoke_query = tokenizer.tokenize(
            [first_lexical_text],
            update_vocab=False,
            return_as="ids",
            show_progress=False,
            allow_empty=False,
        )
        smoke = retriever.retrieve(
            smoke_query,
            k=1,
            show_progress=False,
            backend_selection=args.backend,
        )
        smoke_row = int(smoke.documents[0, 0])
        if not 0 <= smoke_row < row_count:
            raise RuntimeError("BM25S smoke retrieval returned an invalid row")

        elapsed = time.monotonic() - started
        manifest = {
            "format_version": INDEX_FORMAT_VERSION,
            "status": "complete",
            "backend": "bm25s",
            "pipeline": "legal-ir-retrieval-compact",
            "document_key": "context_id",
            "candidate_key": "chunk_id",
            "aggregation": "max_chunk_score_per_context",
            "corpus": str(args.corpus.resolve()),
            "corpus_sha256": None if args.skip_sha256 else sha256_file(args.corpus),
            "chunks": row_count,
            "contexts": len(table.contexts),
            "vocabulary": len(vocab),
            "bm25": {
                "library": "bm25s",
                "library_version": str(getattr(bm25s, "__version__", "unknown")),
                "method": args.method,
                "k1": args.k1,
                "b": args.b,
                "backend": args.backend,
                "dtype": "float32",
            },
            "lexical_profile": args.lexical_profile,
            "word_segmenter": args.word_segmenter,
            "lexical_profile_definition": LEXICAL_PROFILE,
            "lexical_profile_sha256": lexical_profile_signature(
                args.lexical_profile,
                args.word_segmenter,
            ),
            "stemming": False,
            "lemmatization": False,
            "smoke_row": smoke_row,
            "elapsed_seconds": elapsed,
        }
        atomic_write_json(build_dir / "manifest.json", manifest)
        manifest["files"] = index_file_inventory(build_dir)
        atomic_write_json(build_dir / "manifest.json", manifest)
        commit_build_directory(build_dir, args.index_dir, args.recreate)
        if args.report:
            atomic_write_json(args.report, manifest)
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    except Exception:
        discard_build_directory(build_dir)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument(
        "--b",
        type=float,
        default=0.55,
        help="hệ số chuẩn hoá độ dài.",
    )
    parser.add_argument(
        "--method",
        choices=("lucene", "robertson", "atire", "bm25l", "bm25+"),
        default="lucene",
    )
    parser.add_argument("--backend", choices=("numpy", "numba"), default="numpy")
    parser.add_argument(
        "--lexical-profile", choices=("plain", "enhanced"), default="enhanced"
    )
    parser.add_argument(
        "--word-segmenter",
        choices=("simple", "underthesea", "vncorenlp"),
        default="simple",
    )
    parser.add_argument("--vncorenlp-dir", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--progress-every", type=int, default=10_000)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    return parser


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
