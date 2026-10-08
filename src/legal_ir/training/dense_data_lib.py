"""Kiểm tra và ghép dữ liệu tinh chỉnh encoder dense.

File này làm gì
----------------
Kiểm tra các bộ qrels có trỏ tới văn bản thật trong corpus không, điều chỉnh thang điểm
về khoảng [0, 1] cho việc ghép nhiều nguồn, và lọc các cặp train có ứng viên tích cực
nằm trong nhóm đầu.

Pipeline stage
--------------
Không thuộc bước cụ thể nào. Được import bởi các bước huấn luyện dense.

Input / Output
--------------
  Xem từng hàm. Phần lớn nhận danh sách dict và trả về danh sách dict đã lọc.

Model/class
-----------
Không có.

Command
-------
  Không chạy trực tiếp; đây là module được import.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import NamedTuple, Protocol

from legal_ir.core.commons import iter_retrieval_rows


class TrainingQuery(NamedTuple):
    qid: str
    question: str
    gold_context_ids: list[str]


def load_training_queries(path: Path) -> dict[str, TrainingQuery]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        qid: TrainingQuery(qid=qid, question=rec["question"], gold_context_ids=list(rec["answer"]))
        for qid, rec in raw.items()
    }


def verify_corpus_sha256(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Corpus sha256 mismatch for {path}: expected {expected}, got {actual}. "
            "Corpus may be stale — do not proceed without re-verifying provenance."
        )


def group_chunks_for_contexts(
    corpus_path: Path, required_context_ids: set[str]
) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in iter_retrieval_rows(corpus_path):
        cid = row["context_id"]
        if cid in required_context_ids:
            grouped.setdefault(cid, []).append({"chunk_id": row["chunk_id"], "text": row["text"]})
    return grouped


class ChunkScorer(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class PositiveCandidate(NamedTuple):
    chunk_id: str
    text: str
    teacher_score: float
    selected_rank: int


def select_positives(
    query: str, chunks: list[dict], scorer: ChunkScorer, margin: float = 0.05
) -> list[PositiveCandidate]:
    texts = [c["text"] for c in chunks]
    scores = scorer.score(query, texts)
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)

    top_chunk, top_score = ranked[0]
    result = [
        PositiveCandidate(
            chunk_id=top_chunk["chunk_id"], text=top_chunk["text"],
            teacher_score=top_score, selected_rank=1,
        )
    ]
    if len(ranked) > 1:
        second_chunk, second_score = ranked[1]
        if (top_score - second_score) <= margin:
            result.append(
                PositiveCandidate(
                    chunk_id=second_chunk["chunk_id"], text=second_chunk["text"],
                    teacher_score=second_score, selected_rank=2,
                )
            )
    return result


def validate_qrels_resolve(
    queries: dict[str, TrainingQuery], corpus_context_ids: set[str]
) -> None:
    missing: list[tuple[str, str]] = []
    for q in queries.values():
        for cid in q.gold_context_ids:
            if cid not in corpus_context_ids:
                missing.append((q.qid, cid))
    if missing:
        raise ValueError(f"Qrels do not resolve to corpus, {len(missing)} missing pairs: {missing[:10]}")


def validate_positive_in_context(
    positive: PositiveCandidate, context_id: str, chunks_for_context: list[dict]
) -> None:
    valid_ids = {c["chunk_id"] for c in chunks_for_context}
    if positive.chunk_id not in valid_ids:
        raise ValueError(
            f"positive_chunk_id {positive.chunk_id!r} does not belong to declared "
            f"context_id {context_id!r}"
        )


def validate_teacher_score_range(score: float) -> None:
    if math.isnan(score):
        raise ValueError("Teacher score is NaN")
    if not (-10.0 <= score <= 10.0):
        raise ValueError(f"Teacher score {score} outside expected range [-10, 10]")


def validate_no_duplicate_positive(qid: str, chunk_id: str, seen: set[tuple[str, str]]) -> None:
    key = (qid, chunk_id)
    if key in seen:
        raise ValueError(f"Duplicate (qid, positive_chunk_id): {key}")
    seen.add(key)


def filter_nonempty_text(text: str) -> bool:
    return bool(text.strip())
