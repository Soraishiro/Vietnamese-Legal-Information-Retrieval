"""Chuẩn bị dữ liệu tinh chỉnh encoder dense.

File này làm gì
----------------
Ba lệnh: `split` chia tập huấn luyện thành phần train và phần dev; `pool` dựng tập đánh
giá gồm ứng viên tích cực có nhãn và ứng viên tiêu cực khó; `filter` lọc cặp train để dựng
dữ liệu tinh chỉnh, loại các câu đã nằm trong phần dev.

Pipeline stage
--------------
Ngoài đường suy luận. Chỉ chạy khi muốn dựng lại checkpoint từ đầu.

Input
-----
  --train-json  file nhãn train: {qid: {"question": ..., "answer": [...]}}
  --qrels       file qrels train
  --corpus      retrieval_chunks.jsonl (cho lệnh `pool`)
  --sources     các ranked.jsonl dùng làm nguồn ứng viên tiêu cực khó (cho lệnh `pool`)
  --out         file kết quả của lệnh đang chạy

Output
-------
  dev_split_qids.json, eval_pool.jsonl, hoặc bộ triplet đã lọc, tuỳ lệnh

Model/class
-----------
  Lệnh `pool` dùng một cross-encoder làm thầy để chấm
  ứng viên, và vector hoá bằng `FlagEmbedding`.

Command
-------
  python train_dense_embedder.py split --train-json train.json --qrels <qrels.jsonl> \
      --out dev_split_qids.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from legal_ir.training import dense_format_converters as dfc
from legal_ir.training import dense_mining_lib as dml
from legal_ir.training.dense_data_lib import load_training_queries

_HOST_ACCOUNT = "host"
_DENSE_DATA_DIR = Path(
    "/path/to/project/retrieval/local/training_data/dense"
)


def _iter_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


_YES_NO_PATTERNS = [r"\bkhông\s*\?", r"\bhay\s+không\b", r"\bcó\b.*\bkhông\b"]
_WHEN_PATTERNS = [r"khi nào", r"bao giờ", r"lúc nào", r"thời điểm nào", r"ngày nào", r"năm nào", r"tháng nào"]
_WHERE_PATTERNS = [r"ở đâu", r"tại đâu", r"nơi nào", r"địa điểm nào"]
_WHO_PATTERNS = [r"\bai\b", r"người nào"]
_HOW_PATTERNS = [r"làm sao", r"như thế nào", r"thế nào", r"cách nào", r"bằng cách nào"]

_WHICH_PATTERNS = [
    r"loại\s+.{0,20}?nào", r"kiểu\s+.{0,20}?nào", r"hình thức\s+.{0,20}?nào",
    r"cơ quan\s+.{0,20}?nào", r"tổ chức\s+.{0,20}?nào", r"văn bản\s+.{0,20}?nào",
    r"mức\s+.{0,20}?nào", r"cấp\s+.{0,20}?nào",
]
_WHAT_PATTERNS = [r"là gì", r"\bgì\b", r"\bnào\b"]

_QUESTION_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("yes_no", _YES_NO_PATTERNS),
    ("when", _WHEN_PATTERNS),
    ("where", _WHERE_PATTERNS),
    ("who", _WHO_PATTERNS),
    ("how", _HOW_PATTERNS),
    ("which", _WHICH_PATTERNS),
    ("what", _WHAT_PATTERNS),
]



def classify_question_type(question: str) -> str:
    lowered = question.lower()
    for bucket, patterns in _QUESTION_TYPE_RULES:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return bucket
    return "what"


def n_gold_bucket(n_gold: int) -> str:
    if n_gold <= 0:
        raise ValueError(f"n_gold phai >= 1 (cau hoi khong co gold context nao la du lieu loi): {n_gold}")
    return "single" if n_gold == 1 else "multi"


def stratify_label(question_type: str, gold_bucket: str) -> str:
    return f"{question_type}__{gold_bucket}"


def count_per_label(labels: Sequence[str]) -> dict[str, int]:
    return dict(Counter(labels))


def assert_no_leakage(train_qids: Sequence[str], dev_qids: Sequence[str]) -> None:
    overlap = set(train_qids) & set(dev_qids)
    if overlap:
        raise ValueError(
            f"Dev leakage detected: {len(overlap)} qid xuat hien ca train va dev "
            f"({sorted(overlap)[:10]}...)"
        )


def _preflight_check_label_counts(labels: Sequence[str], min_count: int = 2) -> None:
    counts = count_per_label(labels)
    too_small = {label: n for label, n in counts.items() if n < min_count}
    if too_small:
        raise ValueError(
            f"Stratify label thieu mau (can >= {min_count} de chia ca train/dev): "
            f"{too_small}. Giam so bucket (vd gop question_type hiem) hoac tang "
            f"kich thuoc mau truoc khi chia."
        )


def build_dev_split(
    qids: Sequence[str], labels: Sequence[str], dev_size: int, seed: int
) -> tuple[list[str], list[str]]:
    _preflight_check_label_counts(labels, min_count=2)
    qids_arr = np.asarray(qids)
    labels_arr = np.asarray(labels)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=dev_size, random_state=seed)
    train_idx, dev_idx = next(splitter.split(np.zeros(len(qids_arr)), labels_arr))
    train_qids = list(qids_arr[train_idx])
    dev_qids = list(qids_arr[dev_idx])
    assert_no_leakage(train_qids, dev_qids)
    return train_qids, dev_qids


_SPLIT_DEFAULT_TRAIN_JSON = Path(
    "/path/to/project/data/train.json"
)
_SPLIT_DEFAULT_QRELS = Path(
    "/path/to/project/preprocess/workspace/"
    "preprocess_compact_standard/qrels/train.jsonl"
)
_SPLIT_DEFAULT_OUT = _DENSE_DATA_DIR / "dev_split_qids.json"


def _load_n_gold(qrels_path: Path) -> dict[str, int]:
    n_gold: dict[str, int] = defaultdict(int)
    with qrels_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_gold[str(row["query_id"])] += 1
    return dict(n_gold)


def run_split(
    train_json_path: Path,
    qrels_path: Path,
    dev_size: int,
    seed: int,
    out_path: Path,
    command: str = "unknown",
) -> None:
    queries = load_training_queries(train_json_path)
    n_gold = _load_n_gold(qrels_path)

    qids = list(queries.keys())
    question_types = [classify_question_type(queries[qid].question) for qid in qids]
    gold_buckets = [
        n_gold_bucket(n_gold.get(qid, len(queries[qid].gold_context_ids))) for qid in qids
    ]
    fine_labels = [stratify_label(qt, gb) for qt, gb in zip(question_types, gold_buckets)]


    fine_counts = count_per_label(fine_labels)
    question_types_needing_coarsen = set()
    for qt in set(question_types):
        sub_counts = [fine_counts.get(stratify_label(qt, gb), 0) for gb in ("single", "multi")]
        if any(0 < c < 2 for c in sub_counts):
            question_types_needing_coarsen.add(qt)
    labels = [
        qt if qt in question_types_needing_coarsen else fine_label
        for qt, fine_label in zip(question_types, fine_labels)
    ]
    if question_types_needing_coarsen:
        print(
            f"[split] coarsened (bo n_gold sub-bucket) cho question_type qua hiem: "
            f"{sorted(question_types_needing_coarsen)}",
            flush=True,
        )

    counts_before = count_per_label(labels)
    print(f"[split] {len(qids)} qids, {len(counts_before)} stratify cells", flush=True)
    for label, count in sorted(counts_before.items()):
        print(f"  {label}: {count}", flush=True)

    train_qids, dev_qids = build_dev_split(qids, labels, dev_size=dev_size, seed=seed)

    label_by_qid = dict(zip(qids, labels))
    counts_train = count_per_label([label_by_qid[q] for q in train_qids])
    counts_dev = count_per_label([label_by_qid[q] for q in dev_qids])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"train_qids": train_qids, "dev_qids": dev_qids}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta = {
        "pipeline_stage": "dense-embedder-dev-split-phase3",
        "host_account": _HOST_ACCOUNT,
        "script": "training/train_dense_embedder.py split",
        "command": command,
        "input": {"train_json": str(train_json_path), "qrels": str(qrels_path)},
        "output": str(out_path),
        "seed": seed,
        "dev_size": dev_size,
        "n_train_qids": len(train_qids),
        "n_dev_qids": len(dev_qids),
        "label_counts_before_split": counts_before,
        "label_counts_train": counts_train,
        "label_counts_dev": counts_dev,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[split] wrote {len(train_qids)} train / {len(dev_qids)} dev qids to {out_path}", flush=True)


_FILTER_DEFAULT_NEGATIVES = _DENSE_DATA_DIR / "dense_evidence_train_v1.negatives.jsonl"
_FILTER_DEFAULT_DEV_SPLIT = _DENSE_DATA_DIR / "dev_split_qids.json"
_FILTER_DEFAULT_OUT_NEGATIVES = _DENSE_DATA_DIR / "dense_evidence_train_v1.negatives.trainonly.jsonl"
_FILTER_DEFAULT_OUT_TRIPLETS = _DENSE_DATA_DIR / "dense_evidence_train_v1.triplets.jsonl"
_FILTER_DEFAULT_OUT_FLAGEMBEDDING = _DENSE_DATA_DIR / "dense_evidence_train_v1.flagembedding.jsonl"


def run_filter(
    negatives_path: Path,
    dev_split_path: Path,
    out_negatives_path: Path,
    out_triplets_path: Path,
    out_flagembedding_path: Path,
) -> None:
    dev_split = json.loads(dev_split_path.read_text(encoding="utf-8"))
    train_qids = set(dev_split["train_qids"])
    dev_qids = set(dev_split["dev_qids"])

    assert_no_leakage(train_qids, dev_qids)

    records = _iter_jsonl(negatives_path)
    filtered = [r for r in records if r["qid"] in train_qids]

    print(
        f"[filter] {len(records)} record goc -> {len(filtered)} record "
        f"train-only (loai {len(records) - len(filtered)} thuoc dev)",
        flush=True,
    )

    _write_jsonl(out_negatives_path, filtered)

    triplets = dfc.to_sentence_transformers_triplets(filtered)
    _write_jsonl(out_triplets_path, triplets)
    print(
        f"[filter] GHI DE {out_triplets_path} voi {len(triplets)} dong",
        flush=True,
    )

    flagembedding = dfc.to_flagembedding_format(filtered)
    _write_jsonl(out_flagembedding_path, flagembedding)
    print(
        f"[filter] GHI DE {out_flagembedding_path} voi {len(flagembedding)} dong",
        flush=True,
    )


_POOL_DEFAULT_NEGATIVES = _DENSE_DATA_DIR / "dense_evidence_train_v1.negatives.jsonl"
_POOL_DEFAULT_DEV_SPLIT = _DENSE_DATA_DIR / "dev_split_qids.json"
_POOL_DEFAULT_CORPUS_CONTEXT_CHUNK = _DENSE_DATA_DIR / "corpus_context_chunk.jsonl"
_POOL_DEFAULT_CORPUS = Path(
    "/path/to/project/preprocess/workspace/"
    "preprocess_compact_standard/corpus/retrieval_chunks.jsonl"
)
_POOL_DEFAULT_OUT = _DENSE_DATA_DIR / "eval_pool.jsonl"


def run_build_pool(
    negatives_path: Path,
    dev_split_path: Path,
    corpus_context_chunk_path: Path,
    corpus_path: Path,
    n_random_background: int,
    seed: int,
    out_path: Path,
) -> None:
    dev_qids = set(json.loads(dev_split_path.read_text(encoding="utf-8"))["dev_qids"])
    records = [r for r in _iter_jsonl(negatives_path) if r["qid"] in dev_qids]
    print(f"[build-pool] {len(records)} record cua {len(dev_qids)} dev qid", flush=True)

    pool: dict[str, dict] = {}
    for r in records:
        pool[r["positive_chunk_id"]] = {
            "chunk_id": r["positive_chunk_id"],
            "context_id": r["context_id"],
            "text": r["positive_text"],
        }
        for n in r["negatives"]:
            if n["kind"] != "hard" or n["text"] is None:
                continue
            pool[n["chunk_id"]] = {
                "chunk_id": n["chunk_id"],
                "context_id": n["context_id"],
                "text": n["text"],
            }
    print(f"[build-pool] {len(pool)} candidate tu gold+hard (dedup theo chunk_id)", flush=True)

    corpus_context_to_chunk = {
        row["context_id"]: row["chunk_id"] for row in _iter_jsonl(corpus_context_chunk_path)
    }
    exclude_context_ids = {c["context_id"] for c in pool.values()}
    rng = random.Random(seed)
    random_hits = dml.sample_random_negatives(
        corpus_context_to_chunk, exclude_context_ids, n_random_background, rng
    )
    random_chunk_ids = {h["chunk_id"] for h in random_hits}
    random_texts = dml.lookup_chunk_texts(corpus_path, random_chunk_ids)
    for h in random_hits:
        pool[h["chunk_id"]] = {
            "chunk_id": h["chunk_id"],
            "context_id": h["context_id"],
            "text": random_texts[h["chunk_id"]],
        }
    print(
        f"[build-pool] +{len(random_hits)} random background -> {len(pool)} candidate tong",
        flush=True,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in pool.values():
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[build-pool] wrote {len(pool)} rows to {out_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    p_split = subparsers.add_parser("split", help="Build stratified dev split tu train.json")
    p_split.add_argument("--train-json", type=Path, default=_SPLIT_DEFAULT_TRAIN_JSON)
    p_split.add_argument("--qrels", type=Path, default=_SPLIT_DEFAULT_QRELS)
    p_split.add_argument("--dev-size", type=int, default=500)
    p_split.add_argument("--seed", type=int, default=20260904)
    p_split.add_argument("--out", type=Path, default=_SPLIT_DEFAULT_OUT)

    p_filter = subparsers.add_parser("filter", help="Loc negatives.jsonl chi con train_qids")
    p_filter.add_argument("--negatives", type=Path, default=_FILTER_DEFAULT_NEGATIVES)
    p_filter.add_argument("--dev-split", type=Path, default=_FILTER_DEFAULT_DEV_SPLIT)
    p_filter.add_argument("--out-negatives", type=Path, default=_FILTER_DEFAULT_OUT_NEGATIVES)
    p_filter.add_argument("--out-triplets", type=Path, default=_FILTER_DEFAULT_OUT_TRIPLETS)
    p_filter.add_argument("--out-flagembedding", type=Path, default=_FILTER_DEFAULT_OUT_FLAGEMBEDDING)

    p_pool = subparsers.add_parser("build-pool", help="Build eval_pool.jsonl query-specific")
    p_pool.add_argument("--negatives", type=Path, default=_POOL_DEFAULT_NEGATIVES)
    p_pool.add_argument("--dev-split", type=Path, default=_POOL_DEFAULT_DEV_SPLIT)
    p_pool.add_argument("--corpus-context-chunk", type=Path, default=_POOL_DEFAULT_CORPUS_CONTEXT_CHUNK)
    p_pool.add_argument("--corpus", type=Path, default=_POOL_DEFAULT_CORPUS)
    p_pool.add_argument("--n-random-background", type=int, default=3000)
    p_pool.add_argument("--seed", type=int, default=20260904)
    p_pool.add_argument("--out", type=Path, default=_POOL_DEFAULT_OUT)

    args = parser.parse_args(argv)

    if args.subcommand == "split":
        run_split(
            train_json_path=args.train_json, qrels_path=args.qrels, dev_size=args.dev_size,
            seed=args.seed, out_path=args.out, command=" ".join(sys.argv),
        )
        return 0
    if args.subcommand == "filter":
        run_filter(
            negatives_path=args.negatives, dev_split_path=args.dev_split,
            out_negatives_path=args.out_negatives, out_triplets_path=args.out_triplets,
            out_flagembedding_path=args.out_flagembedding,
        )
        return 0
    if args.subcommand == "build-pool":
        run_build_pool(
            negatives_path=args.negatives, dev_split_path=args.dev_split,
            corpus_context_chunk_path=args.corpus_context_chunk, corpus_path=args.corpus,
            n_random_background=args.n_random_background, seed=args.seed, out_path=args.out,
        )
        return 0
    raise ValueError(f"subcommand khong duoc ho tro: {args.subcommand!r}")


if __name__ == "__main__":
    raise SystemExit(main())
