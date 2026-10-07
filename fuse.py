"""Gộp xếp hạng của nhiều retriever thành một bảng xếp hạng theo RRF.

File này làm gì
----------------
Mỗi retriever cho ra một danh sách context xếp hạng riêng. File này gộp chúng: với mỗi
context, điểm là tổng `w_i / (rrf_k + rank_i)` trên các nguồn có xếp hạng. Context không có
hạng ở nguồn nào thì bỏ qua số hạng đó, không coi là 0.

Pipeline stage
--------------
Sau khi đã chạy xong toàn bộ retriever (BM25 + các retriever dense), trước rerank.

Input
-----
  --source (lặp lại)   NAME=đường/dẫn/ranked.jsonl, tên nguồn dùng làm tiền tố trường
  --weights (tuỳ chọn) trọng số mỗi nguồn; mặc định đều bằng nhau
  --rrf-k (tuỳ chọn)   hằng số làm mềm; mặc định 60
  --pool (tuỳ chọn)    số ứng viên giữ lại mỗi câu; mặc định 200
  --output             thư mục ghi kết quả

Output
-------
  <output>/ranked.jsonl   mỗi dòng một câu: query_id, ranked (danh sách context_id),
                          hits (hạng của context đó ở từng nguồn), thang điểm fusion
  <output>/report.json    số câu, kích thước pool, trọng số, rrf_k, sha256 của ranked.jsonl

Model/class
-----------
Không có.

Command
-------
  python fuse.py --source bm25=... --source vae=... --source bgem3=... --source f2llm=... \
      --output <out>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from commons import (
    atomic_write_json,
    atomic_write_jsonl,
    build_submission,
    iter_jsonl,
    numeric_sort_key,
)

FUSION_WEIGHTS = {
    "bm25": 0.20569207290714128,
    "vae": 0.3981416769041782,
    "bgem3": 0.15407202701928358,
    "f2llm": 0.2420942231693969,
}
FUSION_RRF_K = 15.0
CANDIDATE_DEPTH = 200


def parse_sources(raw_pairs: Sequence[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for pair in raw_pairs:
        if "=" not in pair:
            raise ValueError(
                f"--source must be name=path, got {pair!r} "
                "(example: --source bm25=path/to/bm25_ranked.jsonl)"
            )
        name, _, path = pair.partition("=")
        name = name.strip()
        if not name or name in sources:
            raise ValueError(f"Invalid or duplicate source name: {name!r}")
        sources[name] = Path(path)
    if len(sources) < 2:
        raise ValueError("At least two --source rankings are required")
    return sources


def load_rankings(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    order: list[str] = []
    output: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(iter_jsonl(path), 1):
        query_id = str(row.get("query_id", ""))
        ranked = row.get("ranked")
        if not query_id or not isinstance(ranked, list):
            raise ValueError(f"Invalid ranked row at {path}:{row_number}")
        if query_id in output:
            raise ValueError(f"Duplicate query_id in {path}: {query_id}")
        contexts = list(dict.fromkeys(str(item) for item in ranked if str(item)))
        evidence: dict[str, dict[str, Any]] = {}
        for hit in row.get("hits", []):
            if not isinstance(hit, Mapping):
                continue
            context_id = str(hit.get("context_id", ""))
            if context_id and context_id not in evidence:
                evidence[context_id] = dict(hit)
        order.append(query_id)
        output[query_id] = {
            "question": str(row.get("question", "")),
            "ranked": contexts,
            "evidence": evidence,
        }
    if not order:
        raise ValueError(f"Ranked file is empty: {path}")
    return order, output


def attach_chunk_evidence(
    hit: dict[str, Any],
    evidence_by_source: Mapping[str, Mapping[str, Mapping[str, Any]]],
    source_order: Sequence[str],
) -> dict[str, Any]:
    enriched = dict(hit)
    ranked_sources = [
        (source, int(hit[f"{source}_rank"]))
        for source in source_order
        if hit.get(f"{source}_rank") is not None
    ]
    ranked_sources.sort(key=lambda pair: pair[1])
    for source, _ in ranked_sources:
        source_hit = evidence_by_source.get(source, {}).get(hit["context_id"], {})
        if "chunk_id" in source_hit:
            enriched[f"{source}_chunk_id"] = source_hit["chunk_id"]
        if "raw_score" in source_hit:
            enriched[f"{source}_raw_score"] = source_hit["raw_score"]
    if ranked_sources:
        best_source = ranked_sources[0][0]
        best = evidence_by_source.get(best_source, {}).get(hit["context_id"], {})
        for key in ("chunk_id", "view", "row_id", "raw_score"):
            if key in best:
                enriched.setdefault(key, best[key])
    return enriched


def submission_from_rankings(
    query_order: Sequence[str],
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    top_k: int,
) -> dict[str, dict[str, list[str]]]:
    answers = {
        query_id: [str(hit["context_id"]) for hit in rankings[query_id][:top_k]]
        for query_id in query_order
    }
    return build_submission(query_order, answers)


def load_all(
    sources: Mapping[str, Path],
) -> tuple[list[str], dict[str, Mapping[str, Mapping[str, Any]]]]:
    order: list[str] | None = None
    loaded: dict[str, Mapping[str, Mapping[str, Any]]] = {}
    for name, path in sources.items():
        current_order, rankings = load_rankings(path)
        if order is None:
            order = current_order
        elif set(order) != set(current_order):
            raise ValueError(
                f"Ranked file {name} has different query IDs than the first source"
            )
        loaded[name] = rankings
    assert order is not None
    for query_id in order:
        if not any(loaded[name][query_id]["ranked"] for name in sources):
            raise ValueError(f"Query {query_id} has an empty ranking in all sources")
    return order, loaded


def fuse_one(
    rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float],
    rrf_k: float,
    candidate_depth: int,
    evidence_by_source: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    scores: dict[str, float] = {}
    evidence: dict[str, dict[str, int | None]] = {}
    for source, ranking in rankings.items():
        weight = float(weights[source])
        if weight <= 0.0:
            continue
        for rank, context_id in enumerate(ranking[:candidate_depth], 1):
            scores[context_id] = scores.get(context_id, 0.0) + weight / (rrf_k + rank)
            evidence.setdefault(context_id, {})[f"{source}_rank"] = rank
    source_order = tuple(rankings)
    ranked = [
        attach_chunk_evidence(
            {"context_id": context_id, "score": score, **evidence[context_id]},
            evidence_by_source,
            source_order,
        )
        for context_id, score in scores.items()
    ]
    ranked.sort(
        key=lambda item: (
            -float(item["score"]),
            numeric_sort_key(item["context_id"]),
        )
    )
    return ranked


def fuse_all(
    query_order: Sequence[str],
    loaded: Mapping[str, Mapping[str, Mapping[str, Any]]],
    weights: Mapping[str, float],
    rrf_k: float,
    candidate_depth: int,
) -> dict[str, list[dict[str, Any]]]:
    return {
        query_id: fuse_one(
            {name: loaded[name][query_id]["ranked"] for name in loaded},
            weights,
            rrf_k,
            candidate_depth,
            {
                name: loaded[name][query_id].get("evidence", {})
                for name in loaded
            },
        )
        for query_id in query_order
    }


def run(args: argparse.Namespace) -> int:
    if not 1 <= args.top_k <= 5:
        raise ValueError("The official evaluator allows at most five answers")

    sources = parse_sources(args.source)
    source_names = list(sources.keys())
    if set(source_names) != set(FUSION_WEIGHTS):
        raise ValueError(
            f"--source names must be exactly {sorted(FUSION_WEIGHTS)}, "
            f"got {sorted(source_names)}"
        )
    order, loaded = load_all(sources)

    weights = dict(FUSION_WEIGHTS)
    rrf_k = FUSION_RRF_K
    candidate_depth = CANDIDATE_DEPTH

    rankings = fuse_all(order, loaded, weights, rrf_k, candidate_depth)
    submission = submission_from_rankings(order, rankings, args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = args.output_dir / "submission.json"
    ranked_path = args.output_dir / "ranked.jsonl"
    atomic_write_json(submission_path, submission)
    atomic_write_jsonl(
        ranked_path,
        (
            {
                "query_id": query_id,
                "question": next(
                    (
                        loaded[name][query_id]["question"]
                        for name in source_names
                        if loaded[name][query_id]["question"]
                    ),
                    "",
                ),
                "method": "weighted-rrf-nmodel",
                "ranked": [hit["context_id"] for hit in rankings[query_id]],
                "hits": (
                    rankings[query_id]
                    if not args.debug_top
                    else rankings[query_id][: args.debug_top]
                ),
            }
            for query_id in order
        ),
    )

    report: dict[str, Any] = {
        "method": "weighted-reciprocal-rank-fusion-nmodel",
        "sources": {
            name: str(sources[name].resolve()) for name in source_names
        },
        "n_queries": len(order),
        "top_k": args.top_k,
        "candidate_depth": candidate_depth,
        **{f"w_{name}": weights[name] for name in source_names},
        "rrf_k": rrf_k,
        "submission": str(submission_path.resolve()),
        "ranked": str(ranked_path.resolve()),
    }
    atomic_write_json(args.output_dir / "metrics.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="ranked.jsonl of one source; repeat for each source (>= 2)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
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
