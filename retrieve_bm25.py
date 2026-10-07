"""Truy hồi BM25 trên corpus đã dựng.

File này làm gì
----------------
Nạp chỉ mục BM25 đã dựng sẵn ở bước trước, truy hồi theo từng câu hỏi, ghi ra
`ranked.jsonl` đúng định dạng mà bước fusion cần: mỗi dòng một câu, kèm hạng của từng
context trong corpus.

Pipeline stage
--------------
Sau khi dựng chỉ mục BM25, cùng nhóm với các retriever dense.

Input
-----
  --index-dir  thư mục chỉ mục BM25
  --queries    file câu hỏi cần truy hồi
  --output-dir thư mục ghi kết quả
  --top-k      số kết quả mỗi câu

Output
-------
  <output-dir>/ranked.jsonl    mỗi dòng: query_id, ranked, method, thang điểm
  <output-dir>/report.json     tham số truy hồi, phiên bản thư viện, sha256

Model/class
-----------
Không dùng mô hình neural. Dùng bm25s, chuẩn Lucene với b=0.55, backend numpy.

Command
-------
  python retrieve_bm25.py --index-dir <R>/indexes/bm25 --queries <W>/queries/private.jsonl \
      --output-dir <R>/outputs/bm25
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from index_bm25 import whitespace_splitter
from commons import (
    aggregate_chunk_hits,
    atomic_write_json,
    atomic_write_jsonl,
    build_word_segmenter,
    build_submission,
    fill_with_fallback,
    lexical_profile_signature,
    lexical_query_tokens,
    lexical_tokens_to_text,
    load_chunk_table,
    load_index_manifest,
    load_queries,
    select_answer_ids,
    validate_view_weights,
)


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


def _tokenize_queries(
    tokenizer: Any,
    questions: Sequence[str],
    profile: str,
    segmenter: Any,
):
    lexical = [
        lexical_tokens_to_text(
            lexical_query_tokens(question, profile=profile, segmenter=segmenter)
        )
        for question in questions
    ]
    return tokenizer.tokenize(
        lexical,
        update_vocab="never",
        return_as="ids",
        show_progress=False,
        allow_empty=True,
    )


def run(args: argparse.Namespace) -> int:
    if not args.queries.is_file():
        raise FileNotFoundError(args.queries)
    if not args.index_dir.is_dir():
        raise FileNotFoundError(args.index_dir)
    if not 1 <= args.top_k <= 5:
        raise ValueError("The official evaluator allows at most five answers")
    if args.context_pool < args.top_k:
        raise ValueError("context-pool must be >= top-k")
    if args.chunk_depth < args.context_pool:
        raise ValueError("chunk-depth must be >= context-pool")
    if args.max_chunk_depth < args.chunk_depth:
        raise ValueError("max-chunk-depth must be >= chunk-depth")
    if args.query_batch_size <= 0 or args.threads < -1:
        raise ValueError("Invalid query-batch-size or threads")
    if not 0.0 <= args.score_ratio_cutoff <= 1.0:
        raise ValueError("score-ratio-cutoff must be between 0 and 1")
    weights = view_weights(args)

    try:
        import bm25s
        from bm25s.tokenization import Tokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Missing BM25 dependencies. Install requirements.txt before retrieval."
        ) from exc

    manifest = load_index_manifest(args.index_dir, backend="bm25s")
    profile = str(manifest.get("lexical_profile", ""))
    word_segmenter = str(manifest.get("word_segmenter", "simple"))
    if lexical_profile_signature(profile, word_segmenter) != manifest.get(
        "lexical_profile_sha256"
    ):
        raise RuntimeError(
            "The runtime lexical analyzer differs from the analyzer used to build the index; "
            "rebuild the BM25 index with this code version."
        )
    queries = load_queries(args.queries)
    if not queries:
        raise ValueError(f"Query file is empty: {args.queries}")

    backend = (
        str(manifest.get("bm25", {}).get("backend", "numpy"))
        if args.backend == "auto"
        else args.backend
    )
    retriever = bm25s.BM25.load(
        args.index_dir,
        mmap=args.mmap,
        load_corpus=False,
        override_params={"backend": backend},
        show_progress=False,
    )
    tokenizer = Tokenizer(
        lower=False,
        splitter=whitespace_splitter,
        stopwords=[],
        stemmer=None,
    )
    tokenizer.load_vocab(str(args.index_dir))
    segmenter = build_word_segmenter(word_segmenter, args.vncorenlp_dir)
    table = load_chunk_table(args.index_dir, mmap=args.mmap)
    indexed_rows = int(retriever.scores["num_docs"])
    if indexed_rows != len(table) or indexed_rows != int(manifest["chunks"]):
        raise RuntimeError(
            f"BM25 index/sidecar count mismatch: bm25={indexed_rows}, "
            f"sidecars={len(table)}, manifest={manifest['chunks']}"
        )

    initial_depth = min(args.chunk_depth, indexed_rows)
    maximum_depth = min(args.max_chunk_depth, indexed_rows)
    rankings: dict[str, list[dict[str, Any]]] = {}
    effective_depths: dict[str, int] = {}
    started = time.monotonic()

    def search_one(query_tokens: Any, depth: int) -> tuple[Any, Any]:
        result = retriever.retrieve(
            [query_tokens],
            k=depth,
            show_progress=False,
            n_threads=args.threads,
            backend_selection="auto",
        )
        return result.documents[0], result.scores[0]

    for start in range(0, len(queries), args.query_batch_size):
        batch = queries[start : start + args.query_batch_size]
        query_ids = _tokenize_queries(
            tokenizer,
            [query["question"] for query in batch],
            profile,
            segmenter,
        )
        result = retriever.retrieve(
            query_ids,
            k=initial_depth,
            show_progress=False,
            n_threads=args.threads,
            backend_selection="auto",
        )
        for position, query in enumerate(batch):
            indices = result.documents[position]
            scores = result.scores[position]
            depth = initial_depth
            hits = aggregate_chunk_hits(
                indices,
                scores,
                table,
                context_pool=args.context_pool,
                view_weights=weights,
                skip_nonpositive=True,
            )


            while len(hits) < args.top_k and depth < maximum_depth:
                depth = min(depth * 2, maximum_depth)
                indices, scores = search_one(query_ids[position], depth)
                hits = aggregate_chunk_hits(
                    indices,
                    scores,
                    table,
                    context_pool=args.context_pool,
                    view_weights=weights,
                    skip_nonpositive=True,
                )
            fill_with_fallback(hits, table.contexts, args.top_k)
            rankings[query["query_id"]] = hits
            effective_depths[query["query_id"]] = depth
        print(
            f"Retrieved {min(start + len(batch), len(queries))}/{len(queries)} queries",
            flush=True,
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
    atomic_write_json(submission_path, submission)
    atomic_write_jsonl(
        ranked_path,
        (
            {
                "query_id": query["query_id"],
                "question": query["question"],
                "method": "bm25s",
                "ranked": [hit["context_id"] for hit in rankings[query["query_id"]]],
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
        "method": "bm25s-local",
        "index_dir": str(args.index_dir.resolve()),
        "queries": str(args.queries.resolve()),
        "n_queries": len(queries),
        "top_k": args.top_k,
        "context_pool": args.context_pool,
        "chunk_depth": args.chunk_depth,
        "max_chunk_depth": args.max_chunk_depth,
        "maximum_effective_chunk_depth": max(effective_depths.values()),
        "score_ratio_cutoff": args.score_ratio_cutoff,
        "view_weights": weights,
        "lexical_profile": profile,
        "word_segmenter": word_segmenter,
        "stemming": False,
        "lemmatization": False,
        "bm25": manifest["bm25"],
        "runtime_backend": backend,
        "mmap": args.mmap,
        "elapsed_seconds": elapsed,
        "submission": str(submission_path.resolve()),
        "ranked": str(ranked_path.resolve()),
    }
    atomic_write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--context-pool", type=int, default=200)
    parser.add_argument("--chunk-depth", type=int, default=4096)
    parser.add_argument("--max-chunk-depth", type=int, default=32768)
    parser.add_argument("--query-batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument(
        "--backend", choices=("auto", "numpy", "numba"), default="numpy"
    )
    parser.add_argument("--vncorenlp-dir", type=Path)
    parser.add_argument("--mmap", action=argparse.BooleanOptionalAction, default=True)
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
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
