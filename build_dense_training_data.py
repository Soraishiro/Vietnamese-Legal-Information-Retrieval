"""Chọn ứng viên tích cực để tinh chỉnh encoder dense.

File này làm gì
----------------
Với mỗi câu train, chấm các ứng viên bằng một cross-encoder làm thầm, rồi giữ lại ứng viên
được chấm cao nhất trong số các ứng viên có nhãn. Kết quả là danh sách ứng viên tích cực
để đưa vào bước khai thác ứng viên tiêu cực.

Pipeline stage
--------------
Ngoài đường suy luận. Chạy trước `mine_dense_negatives.py`.

Input
-----
  --train-json  file nhãn train
  --corpus      retrieval_chunks.jsonl
  --queries     các ranked.jsonl dùng làm tập ứng viên
  --out         file kết quả

Output
-------
  file jsonl ứng viên tích cực cho từng câu

Model/class
-----------
  Dùng một cross-encoder làm thầy để chấm ứng viên.

Command
-------
  python build_dense_training_data.py --train-json train.json --corpus <corpus> \
      --queries bm25=<...> --out <positives.jsonl>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import dense_data_lib as ddl

HOST_ACCOUNT = "host"


def _collect_positive_records(
    queries: dict[str, ddl.TrainingQuery],
    chunks_by_context: dict[str, list[dict]],
    scorer: ddl.ChunkScorer,
) -> tuple[list[dict], int, int]:
    seen_positives: set[tuple[str, str]] = set()
    n_empty_question_filtered = 0
    n_rank2_included = 0
    records: list[dict] = []
    total = len(queries)

    for i, q in enumerate(queries.values(), start=1):
        if i == 1 or i % 200 == 0 or i == total:
            print(f"[build_dense_training_data] scoring query {i}/{total}", flush=True)
        if not ddl.filter_nonempty_text(q.question):
            n_empty_question_filtered += 1
            continue
        for context_id in q.gold_context_ids:
            chunks = chunks_by_context[context_id]
            for positive in ddl.select_positives(q.question, chunks, scorer):
                ddl.validate_positive_in_context(positive, context_id, chunks)
                ddl.validate_teacher_score_range(positive.teacher_score)
                ddl.validate_no_duplicate_positive(q.qid, positive.chunk_id, seen_positives)
                if positive.selected_rank == 2:
                    n_rank2_included += 1
                records.append(
                    {
                        "qid": q.qid,
                        "query": q.question,
                        "positive_chunk_id": positive.chunk_id,
                        "positive_text": positive.text,
                        "context_id": context_id,
                        "positive_teacher_score": positive.teacher_score,
                        "selected_rank": positive.selected_rank,
                        "context_chunk_count": len(chunks),
                    }
                )
    return records, n_empty_question_filtered, n_rank2_included


def _build_meta(
    train_path: Path,
    corpus_path: Path,
    corpus_sha256: str,
    model_checkpoint: str,
    command: str,
    out_jsonl: Path,
    n_queries_processed: int,
    n_positives_written: int,
    n_empty_question_filtered: int,
    n_rank2_included: int,
) -> dict:
    return {
        "pipeline_stage": "dense-embedder-data-build-phase1",
        "host_account": HOST_ACCOUNT,
        "script": "training/build_dense_training_data.py",
        "command": command,
        "model_checkpoint": model_checkpoint,
        "input": {"train_json": str(train_path), "corpus": str(corpus_path)},
        "output": str(out_jsonl),
        "corpus_sha256": corpus_sha256,
        "seed": None,
        "seed_reason": "Phase 1 has no randomness — teacher scoring + selection are deterministic given a fixed checkpoint",
        "n_queries_processed": n_queries_processed,
        "n_positives_written": n_positives_written,
        "n_empty_question_filtered": n_empty_question_filtered,
        "n_rank2_included": n_rank2_included,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run(
    train_path: Path,
    corpus_path: Path,
    corpus_sha256: str,
    scorer: ddl.ChunkScorer,
    out_jsonl: Path,
    out_meta: Path,
    model_checkpoint: str = "unknown",
    command: str = "unknown",
) -> None:
    ddl.verify_corpus_sha256(corpus_path, expected=corpus_sha256)

    queries = ddl.load_training_queries(train_path)
    required_context_ids = {cid for q in queries.values() for cid in q.gold_context_ids}
    chunks_by_context = ddl.group_chunks_for_contexts(corpus_path, required_context_ids)
    ddl.validate_qrels_resolve(queries, corpus_context_ids=set(chunks_by_context))

    records, n_empty_question_filtered, n_rank2_included = _collect_positive_records(
        queries, chunks_by_context, scorer
    )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    meta = _build_meta(
        train_path, corpus_path, corpus_sha256, model_checkpoint, command, out_jsonl,
        len(queries), len(records), n_empty_question_filtered, n_rank2_included,
    )
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_scorer(checkpoint_path: str) -> ddl.ChunkScorer:
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(checkpoint_path)

    class RerankerChunkScorer:
        def score(self, query: str, texts: list[str]) -> list[float]:
            pairs = [(query, t) for t in texts]
            return [float(s) for s in model.predict(pairs)]

    return RerankerChunkScorer()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--corpus-sha256", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-meta", type=Path, required=True)
    args = parser.parse_args()

    scorer = _build_scorer(args.checkpoint)
    run(
        train_path=args.train_json,
        corpus_path=args.corpus,
        corpus_sha256=args.corpus_sha256,
        scorer=scorer,
        out_jsonl=args.out_jsonl,
        out_meta=args.out_meta,
        model_checkpoint=args.checkpoint,
        command=" ".join(sys.argv),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
