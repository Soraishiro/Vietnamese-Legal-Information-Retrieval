"""Khai thác ứng viên tiêu cực khó cho encoder dense.

File này làm gì
----------------
Từ danh sách ứng viên tích cực đã chọn, lấy các ứng viên đứng đầu trong kết quả truy hồi
làm ứng viên tiêu cực khó, rồi trộn thêm ứng viên ngẫu nhiên và ghi ra file cặp train.

Pipeline stage
--------------
Ngoài đường suy luận. Đầu vào của bước tinh chỉnh encoder dense.

Input
-----
  --positives  file ứng viên tích cực
  --qrels      file qrels train
  --sources    các ranked.jsonl
  --corpus     retrieval_chunks.jsonl
  --out        file kết quả
  --n-hard, --n-random, --seed

Output
-------
  file jsonl, mỗi dòng là một câu kèm ứng viên tích cực và ứng viên tiêu cực

Model/class
-----------
Không có.

Command
-------
  python mine_dense_negatives.py --positives <positives.jsonl> --qrels <qrels.jsonl> \
      --sources bm25=<...> --corpus <corpus> --out <pairs.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

from legal_ir.training import dense_data_lib as ddl
from legal_ir.training import dense_mining_lib as dml

HOST_ACCOUNT = "host"


def _iter_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _mine_picks(
    positives: list[dict],
    qrels: dict[str, set[str]],
    loaded_sources: dict[str, dict[str, dict]],
    corpus_context_to_chunk: dict[str, str],
    n_hard: int,
    n_random: int,
    rng: random.Random,
    perc_threshold: float,
    top_after_filter: int,
    fusion_method: str,
) -> list[dict]:
    picks: list[dict] = []
    for pos in positives:
        qid = pos["qid"]
        gold_context_ids = qrels.get(qid, {pos["context_id"]})
        per_query_sources = {
            name: by_qid[qid] for name, by_qid in loaded_sources.items() if qid in by_qid
        }


        hard_hits = dml.sample_hard_negatives(
            per_query_sources,
            gold_context_ids,
            n_hard=n_hard,
            top_after_filter=top_after_filter,
            perc_threshold=perc_threshold,
            fusion_method=fusion_method,
        )
        random_hits = dml.sample_random_negatives(
            corpus_context_to_chunk,
            exclude_context_ids=gold_context_ids | {h["context_id"] for h in hard_hits},
            n_random=n_random,
            rng=rng,
        )
        picks.append({"positive": pos, "hard_hits": hard_hits, "random_hits": random_hits})
    return picks


def _build_negative_records(picks: list[dict], chunk_texts: dict[str, str]) -> list[dict]:
    out_records: list[dict] = []
    for p in picks:
        negatives = [
            {
                "chunk_id": h["chunk_id"], "text": chunk_texts.get(h["chunk_id"]),
                "context_id": h["context_id"], "source_model": h["source_model"],
                "original_rank": h["original_rank"], "teacher_score": None, "kind": "hard",


                "fusion_method": h.get("fusion_method"),
                "fusion_score": h.get("fusion_score"),
                "source_models": h.get("source_models", [h["source_model"]]),
                "source_scores": h.get("source_scores", {}),
                "normalized_scores": h.get("normalized_scores", {}),
                "eligible_ranks": h.get("eligible_ranks", {}),
            }
            for h in p["hard_hits"]
        ] + [
            {
                "chunk_id": h["chunk_id"], "text": chunk_texts.get(h["chunk_id"]),
                "context_id": h["context_id"], "source_model": "random",
                "original_rank": None, "teacher_score": None, "kind": "random",
            }
            for h in p["random_hits"]
        ]
        out_records.append({**p["positive"], "negatives": negatives})
    return out_records


def _build_meta(
    positives_path: Path, qrels_path: Path, sources: dict[str, Path], corpus_sha256: str | None,
    out_path: Path, seed: int, n_hard: int, n_random: int, perc_threshold: float,
    top_after_filter: int, fusion_method: str, n_examples: int, command: str,
) -> dict:
    return {
        "pipeline_stage": "dense-embedder-negative-mining-phase2",
        "host_account": HOST_ACCOUNT,
        "script": "training/mine_dense_negatives.py",
        "command": command,
        "input": {
            "positives": str(positives_path), "qrels": str(qrels_path),
            "sources": {k: str(v) for k, v in sources.items()},
        },
        "output": str(out_path),
        "corpus_sha256": corpus_sha256,
        "seed": seed,
        "n_hard": n_hard,
        "n_random": n_random,
        "perc_threshold": perc_threshold,
        "top_after_filter": top_after_filter,
        "fusion_method": fusion_method,
        "n_examples": n_examples,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    positives_path: Path,
    qrels_path: Path,
    sources: dict[str, Path],
    corpus_context_chunk_path: Path,
    corpus_path: Path,
    n_hard: int,
    n_random: int,
    seed: int,
    out_path: Path,
    corpus_sha256: str | None = None,
    perc_threshold: float = 0.95,
    top_after_filter: int = 10,
    fusion_method: str = "minmax_wsum",
    command: str = "unknown",
    overwrite: bool = False,
) -> None:
    if corpus_sha256 is not None:
        ddl.verify_corpus_sha256(corpus_path, expected=corpus_sha256)

    positives = _iter_jsonl(positives_path)
    qrels: dict[str, set[str]] = {}
    for row in _iter_jsonl(qrels_path):
        qrels.setdefault(str(row["query_id"]), set()).add(str(row["context_id"]))
    loaded_sources = {
        name: {str(row["query_id"]): row for row in dml.iter_jsonl(str(path))}
        for name, path in sources.items()
    }
    corpus_context_to_chunk = {
        row["context_id"]: row["chunk_id"] for row in _iter_jsonl(corpus_context_chunk_path)
    }

    rng = random.Random(seed)
    picks = _mine_picks(
        positives, qrels, loaded_sources, corpus_context_to_chunk, n_hard, n_random, rng,
        perc_threshold=perc_threshold,
        top_after_filter=top_after_filter,
        fusion_method=fusion_method,
    )
    required_chunk_ids = {h["chunk_id"] for p in picks for h in p["hard_hits"] + p["random_hits"]}
    chunk_texts = dml.lookup_chunk_texts(corpus_path, required_chunk_ids)
    out_records = _build_negative_records(picks, chunk_texts)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} da ton tai - dung --overwrite neu co chu dich ghi de. "
            f"Chan ghi de am tham de khong mat artifact mining variant truoc do."
        )
    out_path.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_records), encoding="utf-8"
    )
    meta = _build_meta(
        positives_path, qrels_path, sources, corpus_sha256, out_path, seed, n_hard, n_random,
        perc_threshold, top_after_filter, fusion_method, len(out_records), command,
    )
    out_path.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positives", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--corpus-context-chunk", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", type=str, default=None)
    parser.add_argument("--n-hard", type=int, default=4)
    parser.add_argument("--n-random", type=int, default=2)
    parser.add_argument("--perc-threshold", type=float, default=0.95)
    parser.add_argument("--top-after-filter", type=int, default=10)
    parser.add_argument(
        "--fusion-method",
        choices=["minmax_wsum", "rrf"],
        required=True,
        help=(
            "minmax_wsum: giữ thang điểm nội bộ từng nguồn. rrf: chỉ dùng hạng."
        ),
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Cho phep ghi de out_path neu da ton tai (mac dinh chan de tranh mat artifact).",
    )
    args = parser.parse_args()

    sources = {}
    for spec in args.source:
        name, _, path = spec.partition("=")
        sources[name] = Path(path)

    run(
        positives_path=args.positives, qrels_path=args.qrels, sources=sources,
        corpus_context_chunk_path=args.corpus_context_chunk, corpus_path=args.corpus,
        corpus_sha256=args.corpus_sha256, n_hard=args.n_hard, n_random=args.n_random,
        perc_threshold=args.perc_threshold, top_after_filter=args.top_after_filter,
        fusion_method=args.fusion_method,
        seed=args.seed, out_path=args.out, overwrite=args.overwrite, command=" ".join(sys.argv),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
