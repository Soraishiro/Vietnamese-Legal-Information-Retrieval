"""Khai thác ứng viên tiêu cực khó cho encoder dense.

File này làm gì
----------------
Chọn ứng viên tiêu cực từ kết quả của nhiều retriever: ưu tiên ứng viên có thang điểm
đã chuẩn hoá cao, rồi trộn thêm ứng viên ngẫu nhiên. Việc lấy mẫu dùng seed cố định nên
chạy lại cho kết quả giống nhau.

Pipeline stage
--------------
Không thuộc bước cụ thể nào. Được import bởi `mine_dense_negatives.py`.

Input / Output
--------------
  Xem từng hàm.

Model/class
-----------
Không có.

Command
-------
  Không chạy trực tiếp; đây là module được import.
"""

from __future__ import annotations

import random
from pathlib import Path

import sys
_RETRIEVE_LOCAL = Path(__file__).resolve().parent.parent
if str(_RETRIEVE_LOCAL) not in sys.path:
    sys.path.insert(0, str(_RETRIEVE_LOCAL))

from commons import iter_retrieval_rows
from mine_reranker_pairs import hit_for, iter_jsonl


def gold_score_in_source(source: dict, gold_context_id: str) -> float | None:
    hit = hit_for(source, gold_context_id)
    return hit["score"] if hit else None


def filter_topk_percpos(
    hits: list[dict],
    gold_score: float,
    perc_threshold: float = 0.95,
    exclude_context_ids: frozenset[str] = frozenset(),
) -> list[dict]:
    threshold = perc_threshold * gold_score
    kept = [
        h for h in hits
        if h["score"] < threshold and h["context_id"] not in exclude_context_ids
    ]
    return sorted(kept, key=lambda h: h["score"], reverse=True)


def _deterministic_gold_score(source: dict, gold_context_ids: set[str]) -> float | None:
    for gid in sorted(gold_context_ids):
        score = gold_score_in_source(source, gid)
        if score is not None:
            return float(score)
    return None


def _minmax_scores(hits: list[dict]) -> dict[str, float]:
    if not hits:
        return {}
    values = [float(hit["score"]) for hit in hits]
    lo, hi = min(values), max(values)
    if hi == lo:
        return {str(hit["context_id"]): 1.0 for hit in hits}
    scale = hi - lo
    return {
        str(hit["context_id"]): (float(hit["score"]) - lo) / scale
        for hit in hits
    }


def sample_hard_negatives(
    sources: dict[str, dict],
    gold_context_ids: set[str],
    n_hard: int = 4,
    top_after_filter: int = 10,
    perc_threshold: float = 0.95,
    fusion_method: str = "minmax_wsum",
) -> list[dict]:
    if fusion_method not in {"minmax_wsum", "rrf"}:
        raise ValueError(
            f"fusion_method={fusion_method!r} khong hop le; "
            "chi ho tro 'minmax_wsum' hoac 'rrf'"
        )
    if n_hard < 0 or top_after_filter <= 0:
        raise ValueError("n_hard phai >= 0 va top_after_filter phai > 0")
    if not 0.0 <= perc_threshold <= 1.0:
        raise ValueError("perc_threshold phai nam trong [0, 1]")


    pooled: dict[str, dict] = {}
    excluded = frozenset(gold_context_ids)

    for source_name, source in sorted(sources.items()):
        gold_score = _deterministic_gold_score(source, gold_context_ids)
        if gold_score is None:


            continue

        filtered = filter_topk_percpos(
            source.get("hits", []), gold_score, perc_threshold, excluded
        )[:top_after_filter]
        if not filtered:
            continue

        normalized = _minmax_scores(filtered)
        ranked_list = source.get("ranked", [])

        for eligible_rank, hit in enumerate(filtered, start=1):
            cid = str(hit["context_id"])
            original_rank = ranked_list.index(cid) + 1 if cid in ranked_list else None
            norm_score = float(normalized[cid])
            contribution = (
                norm_score
                if fusion_method == "minmax_wsum"
                else 1.0 / (60.0 + eligible_rank)
            )

            support = {
                "source_model": source_name,
                "chunk_id": hit["chunk_id"],
                "raw_score": float(hit["score"]),
                "normalized_score": norm_score,
                "eligible_rank": eligible_rank,
                "original_rank": original_rank,
                "contribution": contribution,
            }

            if cid not in pooled:
                pooled[cid] = {
                    "context_id": cid,
                    "fusion_score": 0.0,
                    "supports": [],
                    "primary": support,
                }
            entry = pooled[cid]
            entry["fusion_score"] += contribution
            entry["supports"].append(support)


            primary = entry["primary"]
            if (
                contribution > primary["contribution"]
                or (
                    contribution == primary["contribution"]
                    and source_name < primary["source_model"]
                )
            ):
                entry["primary"] = support

    fused: list[dict] = []
    for cid, entry in pooled.items():
        primary = entry["primary"]
        supports = sorted(entry["supports"], key=lambda x: x["source_model"])
        fused.append(
            {
                "context_id": cid,
                "chunk_id": primary["chunk_id"],


                "score": primary["raw_score"],
                "source_model": primary["source_model"],
                "original_rank": primary["original_rank"],
                "fusion_method": fusion_method,
                "fusion_score": float(entry["fusion_score"]),
                "source_models": [s["source_model"] for s in supports],
                "source_scores": {s["source_model"]: s["raw_score"] for s in supports},
                "normalized_scores": {
                    s["source_model"]: s["normalized_score"] for s in supports
                },
                "eligible_ranks": {
                    s["source_model"]: s["eligible_rank"] for s in supports
                },
            }
        )


    fused.sort(
        key=lambda h: (
            -h["fusion_score"],
            min(h["eligible_ranks"].values()),
            h["context_id"],
        )
    )
    return fused[:n_hard]


def sample_random_negatives(
    corpus_context_to_chunk: dict[str, str],
    exclude_context_ids: set[str],
    n_random: int,
    rng: random.Random,
) -> list[dict]:
    pool = sorted(cid for cid in corpus_context_to_chunk if cid not in exclude_context_ids)
    if len(pool) < n_random:
        raise ValueError(f"pool too small: need {n_random}, have {len(pool)}")
    chosen = rng.sample(pool, n_random)
    return [{"context_id": cid, "chunk_id": corpus_context_to_chunk[cid]} for cid in chosen]


def lookup_chunk_texts(corpus_path: Path, required_chunk_ids: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in iter_retrieval_rows(corpus_path):
        if row["chunk_id"] in required_chunk_ids:
            result[row["chunk_id"]] = row["text"]
    return result
