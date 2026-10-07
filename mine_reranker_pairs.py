"""Khai thác cặp train cho cross-encoder.

File này làm gì
----------------
Từ tập train có nhãn và các kết quả truy hồi, dựng các cặp (câu hỏi, ứng viên tích cực,
ứng viên tiêu cực khó) để huấn luyện cross-encoder.

Pipeline stage
--------------
Ngoài đường suy luận. Đầu vào của bước train cross-encoder.

Input
-----
  --queries  file câu hỏi có nhãn train
  --qrels    file qrels train
  --sources  các ranked.jsonl dùng để chọn ứng viên tiêu cực khó
  --corpus   retrieval_chunks.jsonl
  --out      file cặp kết quả

Output
-------
  file jsonl, mỗi dòng là một câu kèm danh sách ứng viên tích cực và tiêu cực

Model/class
-----------
Không có.

Command
-------
  python mine_reranker_pairs.py --queries train.json --qrels <qrels.jsonl> \
      --sources bm25=<...> --corpus <corpus> --out <pairs.jsonl>
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

META = {
    "mining_model": "sparse_enhanced+dense_bgem3_topk_percpos_chunk",
    "mining_window": {"perc_threshold": 0.95, "top_after_filter": 10},
    "n_neg_per_sample": {"n_hard": 4, "n_random": 2},
}


def iter_jsonl(path: str):
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_sources(sources: list[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for spec in sources:
        name, _, path = spec.partition("=")
        data: dict[str, dict] = {}
        for d in iter_jsonl(path):
            data[str(d["query_id"])] = d
        out[name] = data
        print(f"[load] source {name}: {len(data)} queries from {path}", flush=True)
    return out


def load_corpus(path: str) -> tuple[dict[str, str], list[str], dict[str, str]]:
    text: dict[str, str] = {}
    ctx: dict[str, str] = {}
    ids: list[str] = []
    for d in iter_jsonl(path):
        cid = d.get("chunk_id") or d.get("id")
        text[cid] = d.get("text", d.get("content", ""))
        ctx[cid] = str(d.get("context_id", ""))
        ids.append(cid)
    print(f"[load] corpus: {len(ids)} chunks", flush=True)
    return text, ids, ctx


def hit_for(source: dict, ctx: str):
    for h in source.get("hits", []):
        if h["context_id"] == ctx:
            return h
    return None


def window_after(source: dict, gold_ctx: str, top_after_filter: int):
    ranked = source.get("ranked", [])
    hits = source.get("hits", [])
    try:
        pos = ranked.index(gold_ctx)
    except ValueError:
        return []
    return hits[pos + 1: pos + 1 + top_after_filter]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", required=True)
    ap.add_argument("--qrels", required=True)
    ap.add_argument("--source", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perc-threshold", type=float, default=0.95)
    ap.add_argument("--top-after-filter", type=int, default=10)
    ap.add_argument("--n-hard-per-source", type=int, default=2)
    ap.add_argument("--n-random", type=int, default=2)
    ap.add_argument("--fusion-method", choices=["minmax_wsum", "rrf"], default=None,
                    help="cách chọn hard-negative; bỏ trống thì chỉ dùng cửa sổ hạng.")
    ap.add_argument("--n-hard", type=int, default=4,
                    help="số hard-negative tổng khi dùng --fusion-method (mặc định 4).")
    ap.add_argument("--seed", type=int, default=20260819)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    queries = {str(d["query_id"]): d["question"] for d in iter_jsonl(args.queries)}
    qrels: dict[str, list[str]] = {}
    for d in iter_jsonl(args.qrels):
        qrels.setdefault(str(d["query_id"]), []).append(str(d["context_id"]))
    sources = load_sources(args.source)
    corpus_text, corpus_ids, corpus_ctx = load_corpus(args.corpus)

    meta = {**META, "n_random": args.n_random,
            "n_hard_per_source": args.n_hard_per_source,
            "top_after_filter": args.top_after_filter}
    if args.fusion_method:
        meta["mining_model"] = (
            "sparse_enhanced+dense_bgem3+dense_vae+dense_f2llm_topk_percpos_fusion_chunk"
        )
        meta["mining_window"] = {**META["mining_window"],
                                 "fusion_method": args.fusion_method,
                                 "perc_threshold": args.perc_threshold,
                                 "top_after_filter": args.top_after_filter}
        meta["n_hard"] = args.n_hard

    n_pairs = 0
    n_skipped = 0
    n_leak = 0
    with Path(args.out).open("w", encoding="utf-8") as out:
        for qid, qtext in queries.items():
            golds = qrels.get(qid)
            if not golds:
                continue

            gold_ctx = None
            for src_name, src_data in sources.items():
                src = src_data.get(qid)
                if not src:
                    continue
                ranked = src.get("ranked", [])
                for g in golds:
                    if g in ranked:
                        if gold_ctx is None:
                            gold_ctx = g
                        break
                if gold_ctx is not None:
                    break
            if gold_ctx is None:
                n_skipped += 1
                continue

            pos_doc_id = None
            pos_text = None
            for src_name, src_data in sources.items():
                src = src_data.get(qid)
                if not src:
                    continue
                h = hit_for(src, gold_ctx)
                if h:
                    pos_doc_id, pos_text = h["chunk_id"], corpus_text.get(h["chunk_id"])
                    break
            if not pos_doc_id or not pos_text:
                n_skipped += 1
                continue

            neg_doc_ids: list[str] = []
            neg_texts: list[str] = []
            neg_source_rank: list[dict] = []
            used_ctxs = set(golds)

            if args.fusion_method:


                from dense_mining_lib import sample_hard_negatives
                per_query = {name: by_qid[qid] for name, by_qid in sources.items()
                             if qid in by_qid}
                hard_hits = sample_hard_negatives(
                    per_query, set(golds),
                    n_hard=args.n_hard,
                    top_after_filter=args.top_after_filter,
                    perc_threshold=args.perc_threshold,
                    fusion_method=args.fusion_method,
                )
                if len(hard_hits) < args.n_hard:
                    n_skipped += 1
                    continue
                for h in hard_hits:
                    ch = h["chunk_id"]
                    t = corpus_text.get(ch)
                    if not t:
                        continue
                    neg_doc_ids.append(ch)
                    neg_texts.append(t)
                    neg_source_rank.append({h["source_model"]: h["original_rank"]})
                    used_ctxs.add(h["context_id"])
                min_hard = args.n_hard
            else:

                for src_name, src_data in sources.items():
                    src = src_data.get(qid)
                    if not src:
                        continue
                    taken = 0
                    w_idx = 1
                    for h in window_after(src, gold_ctx, args.top_after_filter):
                        if taken >= args.n_hard_per_source:
                            break
                        ctx = h["context_id"]
                        if ctx in used_ctxs:
                            continue
                        ch = h["chunk_id"]
                        t = corpus_text.get(ch)
                        if not t:
                            continue
                        neg_doc_ids.append(ch)
                        neg_texts.append(t)
                        neg_source_rank.append({src_name: w_idx})
                        used_ctxs.add(ctx)
                        taken += 1
                        w_idx += 1
                min_hard = args.n_hard_per_source * len(sources)


            tries = 0
            while len(neg_doc_ids) < min_hard + args.n_random and tries < 5000:
                tries += 1
                ch = rng.choice(corpus_ids)
                ctx = corpus_ctx.get(ch)
                t = corpus_text.get(ch)
                if ctx is None or t is None or ctx in used_ctxs or ch == pos_doc_id:
                    continue
                neg_doc_ids.append(ch)
                neg_texts.append(t)
                neg_source_rank.append({})
                used_ctxs.add(ctx)

            if len(neg_doc_ids) < min_hard:
                n_skipped += 1
                continue

            pair = {
                "qid": qid,
                "query": qtext,
                "pos": [pos_text],
                "pos_doc_id": pos_doc_id,
                "neg": neg_texts,
                "neg_doc_ids": neg_doc_ids,
                "neg_source_rank": neg_source_rank,
                "mining_model": meta["mining_model"],
                "mining_window": meta["mining_window"],
            }
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_pairs += 1
            if n_pairs % 1000 == 0:
                print(f"[mine] {n_pairs} pairs (skipped {n_skipped})", flush=True)

    print(f"[done] {n_pairs} pairs, skipped {n_skipped}, out={args.out}", flush=True)


if __name__ == "__main__":
    main()
