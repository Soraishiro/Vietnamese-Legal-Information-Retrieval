"""Đổi định dạng dữ liệu tinh chỉnh cho từng thư viện.

File này làm gì
----------------
Chuyển bộ triplet chung sang hai định dạng khác nhau: một định dạng cho FlagEmbedding và
một định dạng cho sentence-transformers.

Pipeline stage
--------------
Không thuộc bước cụ thể nào. Được import bởi `train_dense_embedder.py`.

Input / Output
--------------
  Nhận danh sách dict, trả về danh sách dict theo định dạng đích.

Model/class
-----------
Không có.

Command
-------
  Không chạy trực tiếp; đây là module được import.
"""

from __future__ import annotations


def to_flagembedding_format(records: list[dict]) -> list[dict]:
    return [
        {
            "query": r["query"],
            "pos": [r["positive_text"]],
            "neg": [n["text"] for n in r["negatives"] if n["text"] is not None],
        }
        for r in records
    ]


def to_sentence_transformers_triplets(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for r in records:
        for n in r["negatives"]:
            if n["text"] is None:
                continue
            rows.append({"anchor": r["query"], "positive": r["positive_text"], "negative": n["text"]})
    return rows
