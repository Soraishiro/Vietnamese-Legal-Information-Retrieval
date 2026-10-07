#!/usr/bin/env python3
"""Áp bộ tham số cố định lên fused-ranking + logit cross-encoder để xếp lại top-5.

Metadata
--------
2026-09-30

File này làm gì
----------------
Đọc kết quả fusion nhiều nguồn (rank theo context) + logit cross-encoder theo context, áp hai giai đoạn
cộng tín hiệu (tần suất văn bản xuất hiện trong nhãn train; láng giềng câu hỏi train + đồng xuất hiện
trong nhãn train), một giai đoạn thay ứng viên hạng 5 theo hai luật cố định, rồi một luật theo năm nêu
trong câu hỏi — ghi ra top-200 mỗi câu và submission.json (5 id/câu).

Pipeline stage
--------------
Sau fusion + rerank cross-encoder, trước khi nộp kết quả.

Input
-----
  --fused          ranked.jsonl của fusion nhiều nguồn (có "ranked" và "hits")
  --ce-logits      logit cross-encoder theo context: {query_id, context_id, value}
  --questions      câu hỏi cần xếp hạng: {qid: {"question": "..."}}
  --train-json     nhãn train (có đáp án): {qid: {"question": "...", "answer": [...]}}
  --corpus         retrieval_chunks.jsonl (luật năm đọc dòng [VĂN BẢN]/[SỐ/KÝ HIỆU])
  --out-dir        thư mục ghi kết quả
  --reference-ranking (tuỳ chọn) ranking top-200 đã lưu để so từng câu

Output
------
  <out-dir>/top200.jsonl, <out-dir>/submission.json, <out-dir>/submission.zip,
  <out-dir>/report.json

Model/class
-----------
Không có.

Command
-------
  python apply_ranking_rules.py apply-ranking-rules --fused F --ce-logits C --questions Q --train-json T \
      --corpus K --out-dir O [--reference-ranking X]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def numeric_sort_key(context_id: str):
    try:
        return (0, int(context_id))
    except ValueError:
        return (1, context_id)


BLEND_WEIGHT = 0.5


BLEND_RRF_K = 20


POOL_DEPTH = 200


PRODUCTION_FUSION_WEIGHTS = {
    "bm25": 0.20569207290714128,
    "vae": 0.3981416769041782,
    "bgem3": 0.15407202701928358,
    "f2llm": 0.2420942231693969,
}


PRODUCTION_FUSION_RRF_K = 15


def _fusion_score_from_pool(
    rows: list[dict[str, Any]], weights: dict[str, float], rrf_k: float
) -> dict[str, float]:
    """Σ w_i/(rrf_k+rank_i) trên các nguồn CÓ hạng (nguồn không có rank thì bỏ số hạng)."""
    scores: dict[str, float] = {}
    for row in rows:
        ctx = str(row["context_id"])
        total = 0.0
        for src, w in weights.items():
            rk = row.get(f"{src}_rank")
            if rk is not None:
                total += w / (rrf_k + rk)
        scores[ctx] = total
    return scores


def _rank_from_fusion_scores(pool_ctx: list[str], scores: dict[str, float]) -> dict[str, int]:
    """Hạng 1..n theo (-score, numeric_sort_key)."""
    ordered = sorted(pool_ctx, key=lambda c: (-scores[c], numeric_sort_key(c)))
    return {c: i for i, c in enumerate(ordered, 1)}


def _rank_ce_with_tiebreak(
    pool_ctx: list[str], ce_value: dict[str, float], fusion_rank: dict[str, int]
) -> dict[str, int]:
    """Hạng cross-encoder: (-logit, fusion_rank, numeric_sort_key)."""
    ordered = sorted(pool_ctx, key=lambda c: (-ce_value[c], fusion_rank[c], numeric_sort_key(c)))
    return {c: i for i, c in enumerate(ordered, 1)}


def _blend_full_order(
    pool_ctx: list[str],
    fusion_rank: dict[str, int],
    rank_ce: dict[str, int],
    w: float,
    k: float,
) -> list[str]:
    """Trộn cuối: score=w/(k+fusion_rank)+(1-w)/(k+rank_ce); tie-break=(rank_ce, numeric_sort_key)."""
    ordered = sorted(
        pool_ctx,
        key=lambda c: (
            -(w / (k + fusion_rank[c]) + (1.0 - w) / (k + rank_ce[c])),
            rank_ce[c],
            numeric_sort_key(c),
        ),
    )
    return ordered


def _fusion_score_with_prior(
    rows: list[dict[str, Any]], weights: dict[str, float], rrf_k: float,
    n_train: dict[str, int], a: float, mode: str,
) -> dict[str, float]:
    scores = _fusion_score_from_pool(rows, weights, rrf_k)
    for row in rows:
        ctx = str(row["context_id"])
        n = n_train.get(ctx, 0)
        if mode == "log":
            scores[ctx] += a * math.log(1.0 + n)
        elif mode == "seen_flag":
            scores[ctx] += a * (1.0 if n > 0 else 0.0)
        else:
            raise ValueError(f"mode không hỗ trợ: {mode}")
    return scores


def _regroup_ce_rank_by_margin(
    pool_ctx: list[str], ce_value: dict[str, float], fusion_rank: dict[str, int], margin: float
) -> dict[str, int]:
    """Gộp ứng viên liền kề (theo thứ tự CE gốc) cách nhau ≤margin thành nhóm hoà, xếp lại
    trong nhóm theo fusion_rank tăng dần — margin=0 tái lập đúng rank_ce gốc."""
    ordered = sorted(pool_ctx, key=lambda c: (-ce_value[c], fusion_rank[c], numeric_sort_key(c)))
    groups: list[list[str]] = [[ordered[0]]]
    for prev, cur in zip(ordered, ordered[1:]):
        if ce_value[prev] - ce_value[cur] <= margin:
            groups[-1].append(cur)
        else:
            groups.append([cur])
    new_order: list[str] = []
    for g in groups:
        new_order.extend(sorted(g, key=lambda c: (fusion_rank[c], numeric_sort_key(c))))
    return {c: i for i, c in enumerate(new_order, 1)}


PRIOR_MODE = "seen_flag"


TRAIN_LABELS_JSON = Path("train.json")


QUESTIONS_JSON = Path("questions.json")


STAGE1_PARAMS = {"a": 0.0075, "w": 0.45, "k": 12.0, "margin": 0.5}


COOCCURRENCE_ANCHOR_TOP = 3


def _load_train_labels() -> tuple[list[str], list[str], list[list[str]]]:
    """(qids, questions, gold[context_id]) từ nhãn train."""
    qids: list[str] = []
    questions: list[str] = []
    golds: list[list[str]] = []
    raw = json.loads(TRAIN_LABELS_JSON.read_text(encoding="utf-8"))
    for q, item in raw.items():
        qids.append(str(q))
        questions.append(str(item["question"]))
        golds.append([str(d) for d in item["answer"]])
    return qids, questions, golds


def _build_question_context(query_ids: list[str]) -> SimpleNamespace:
    """sim TF-IDF (1-2 gram, fit trên câu hỏi train) giữa câu cần xếp và câu train; doc→cột; n(d);
    đồ thị đồng-gold cnt[a][b]."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_qids, train_questions, train_gold = _load_train_labels()
    all_questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    with_question = [q for q in query_ids if q in all_questions]
    vec = TfidfVectorizer(lowercase=True, analyzer="word", ngram_range=(1, 2), min_df=1)
    train_matrix = vec.fit_transform(train_questions)
    query_matrix = vec.transform([str(all_questions[q]["question"]) for q in with_question])
    sim = (query_matrix @ train_matrix.T).toarray().astype(np.float32)
    doc_cols: dict[str, list[int]] = {}
    cnt: dict[str, dict[str, int]] = {}
    for col, gold in enumerate(train_gold):
        for d in gold:
            doc_cols.setdefault(d, []).append(col)
        for a in gold:
            for b in gold:
                if a != b:
                    cnt.setdefault(a, {})[b] = cnt.setdefault(a, {}).get(b, 0) + 1
    return SimpleNamespace(
        sim=sim, row_of={q: i for i, q in enumerate(with_question)}, train_gold=train_gold,
        doc_cols={d: np.array(c) for d, c in doc_cols.items()},
        n_doc={d: len(c) for d, c in doc_cols.items()}, cnt=cnt, np=np,
    )


def _neighbor_question_bonus(ctx: SimpleNamespace, qid: str, k: int, tau: float) -> dict[str, float]:
    """{doc: Σ sim} trên k câu train giống nhất có sim ≥ τ. Không có câu hỏi/không qua τ → {}."""
    row = ctx.row_of.get(qid)
    if row is None:
        return {}
    sims = ctx.sim[row]
    top = ctx.np.argpartition(-sims, k - 1)[:k] if k < len(sims) else ctx.np.arange(len(sims))
    out: dict[str, float] = {}
    for j in top:
        s = float(sims[j])
        if s >= tau:
            for d in ctx.train_gold[int(j)]:
                out[d] = out.get(d, 0.0) + s
    return out


def _cooccurrence_bonus(ctx: SimpleNamespace, anchors: list[str]) -> dict[str, float]:
    """{doc: Σ_t cnt(doc,t)/sqrt(n(doc)·n(t))} trên các văn bản neo t."""
    out: dict[str, float] = {}
    for t in anchors:
        nt = ctx.n_doc.get(t)
        if not nt:
            continue
        for d, c in ctx.cnt.get(t, {}).items():
            out[d] = out.get(d, 0.0) + c / math.sqrt(ctx.n_doc[d] * nt)
    return out


def _apply_stage1_stage2(
    pool: dict[str, list[dict[str, Any]]], qids: list[str], n_train: dict[str, int],
    extras: list[tuple[float, dict[str, dict[str, float]]]],
    a: float, w: float, k: float, margin: float,
) -> dict[str, list[str]]:
    """Cộng prior tần suất train vào điểm fusion, cộng thêm bonus (extras: kNN câu hỏi + đồng-gold nếu
    có), rồi trộn với hạng cross-encoder (hoà nhóm nếu margin>0)."""
    out: dict[str, list[str]] = {}
    for qid in qids:
        rows = pool[qid]
        pool_ctx = [str(r["context_id"]) for r in rows]
        scores = _fusion_score_with_prior(rows, PRODUCTION_FUSION_WEIGHTS, PRODUCTION_FUSION_RRF_K, n_train, a, PRIOR_MODE)
        for strength, bonus in extras:
            if strength:
                for d, v in bonus.get(qid, {}).items():
                    if d in scores:
                        scores[d] += strength * v
        fusion_rank = _rank_from_fusion_scores(pool_ctx, scores)
        ce_value = {str(r["context_id"]): float(r["ce_value"]) for r in rows}
        rank_ce = _regroup_ce_rank_by_margin(pool_ctx, ce_value, fusion_rank, margin) if margin > 0 \
            else _rank_ce_with_tiebreak(pool_ctx, ce_value, fusion_rank)
        out[qid] = _blend_full_order(pool_ctx, fusion_rank, rank_ce, w, k)
    return out


def _hash_file(path: Path) -> str:
    """SHA-256 theo byte, đọc streaming để không nạp artifact lớn vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


STAGE3_MIN_QUERY_TOKENS = 8


STAGE3_MAX_RETRIEVER_RANK = 10


STAGE3_MAX_FUSION_RANK = 2


STAGE3_TRAIN_COUNT = 10


STAGE3_CE_GAP = 1.0


def _stage3_promotions(
    qid: str,
    stage2_order: list[str],
    base_order: list[str],
    rows_by_context: dict[str, dict[str, Any]],
    n_train: dict[str, int],
    question: str,
) -> tuple[str | None, str | None]:
    """Trả ứng viên của luật ổn định và luật biên."""
    stable: str | None = None
    if len(question.split()) >= STAGE3_MIN_QUERY_TOKENS:
        stage2_rank = {context_id: rank for rank, context_id in enumerate(stage2_order, 1)}
        for context_id in base_order[:5]:
            row = rows_by_context[context_id]
            branch_ranks = [row.get(f"{name}_rank") for name in ("bm25", "vae", "bgem3", "f2llm")]
            if (
                stage2_rank.get(context_id) in (6, 7)
                and n_train.get(context_id, 0) == 0
                and all(rank is not None and int(rank) <= STAGE3_MAX_RETRIEVER_RANK for rank in branch_ranks)
                and int(row["fusion_rank"]) <= STAGE3_MAX_FUSION_RANK
            ):
                stable = context_id
                break

    boundary: str | None = None
    if len(stage2_order) >= 6:
        candidate, cutoff = stage2_order[5], stage2_order[4]
        if (
            n_train.get(candidate, 0) >= STAGE3_TRAIN_COUNT
            and float(rows_by_context[candidate]["ce_value"]) - float(rows_by_context[cutoff]["ce_value"]) >= STAGE3_CE_GAP
        ):
            boundary = candidate
    return stable, boundary


def _stage3_apply(
    mode: str,
    stage2: dict[str, list[str]],
    base: dict[str, list[str]],
    pool: dict[str, list[dict[str, Any]]],
    n_train: dict[str, int],
    questions: dict[str, str],
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    """Thay đúng hạng 5 bằng một ứng viên. Ưu tiên luật ổn định; mỗi câu chỉ đổi một lần."""
    full: dict[str, list[str]] = {}
    triggered_stable: list[str] = []
    triggered_boundary: list[str] = []
    overlap: list[str] = []
    selected: dict[str, dict[str, str]] = {}
    for qid, order in stage2.items():
        rows = {str(row["context_id"]): row for row in pool[qid]}
        stable, boundary = _stage3_promotions(qid, order, base[qid], rows, n_train, questions.get(qid, ""))
        if stable:
            triggered_stable.append(qid)
        if boundary:
            triggered_boundary.append(qid)
        if stable and boundary:
            overlap.append(qid)
        candidate = stable if mode == "stable" else boundary if mode == "boundary" else (stable or boundary)
        rule = "stable" if candidate == stable and stable is not None else "boundary"
        if candidate is None or candidate in order[:5]:
            full[qid] = list(order)
            continue
        new_top5 = list(order[:4]) + [candidate]
        full[qid] = new_top5 + [context_id for context_id in order if context_id not in new_top5]
        selected[qid] = {"rule": rule, "promoted": candidate, "demoted": order[4]}
    audit = {
        "mode": mode,
        "n_triggered_stable": len(triggered_stable),
        "n_triggered_boundary": len(triggered_boundary),
        "n_trigger_overlap": len(overlap),
        "triggered_stable_qids": sorted(triggered_stable, key=int),
        "triggered_boundary_qids": sorted(triggered_boundary, key=int),
        "trigger_overlap_qids": sorted(overlap, key=int),
        "selected": selected,
    }
    return full, audit


YEAR_RULE_WINDOW = 10


YEAR_RULE_PATTERN = re.compile(r"(?<!\d)(19[5-9]\d|20[0-3]\d)(?!\d)")


def _year_rule_context_years(needed: set[str]) -> dict[str, set[str]]:
    """Năm trong dòng [VĂN BẢN] và [SỐ/KÝ HIỆU] của chunk đầu tiên gặp được của mỗi context."""
    years: dict[str, set[str]] = {}
    with CHUNKS_JSONL.open(encoding="utf-8") as stream:
        for line in stream:
            start = line.find('"context_id":"')
            context_id = line[start + 14 : line.find('"', start + 14)]
            if context_id not in needed or context_id in years:
                continue
            text = json.loads(line)["text"]
            header = " ".join(
                part for part in text.split("\n") if part.startswith("[VĂN BẢN]") or part.startswith("[SỐ/KÝ HIỆU]")
            )
            years[context_id] = set(YEAR_RULE_PATTERN.findall(header))
            if len(years) == len(needed):
                break
    return years


def _year_rule_apply(
    ranking: dict[str, list[str]],
    question_years: dict[str, set[str]],
    context_years: dict[str, set[str]],
    window: int,
) -> tuple[dict[str, list[str]], list[str]]:
    result: dict[str, list[str]] = {}
    touched: list[str] = []
    for qid, order in ranking.items():
        years = question_years.get(qid)
        new_order = list(order)
        if years and not any(context_years.get(c, set()) & years for c in order[:5]):
            for candidate in order[5:window]:
                if context_years.get(candidate, set()) & years:
                    top5 = order[:4] + [candidate]
                    new_order = top5 + [c for c in order if c not in top5]
                    touched.append(qid)
                    break
        result[qid] = new_order
    return result, touched


CHUNKS_JSONL = Path("retrieval_chunks.jsonl")


def _build_candidate_pool(
    fused_path: Path, ce_path: Path
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    """(pool 200 candidate/câu với hạng 4 nguồn + logit CE, base = xếp hạng gốc top-200) từ fused
    ranked.jsonl + logit CE context-max."""
    ce_value: dict[tuple[str, str], float] = {}
    with ce_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                ce_value[(str(row["query_id"]), str(row["context_id"]))] = float(row["value"])
    pool: dict[str, list[dict[str, Any]]] = {}
    base: dict[str, list[str]] = {}
    with fused_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["query_id"])
            pool_ctx = [str(c) for c in row["ranked"][:POOL_DEPTH]]
            fusion_rank = {c: i for i, c in enumerate(pool_ctx, 1)}
            hits_by_ctx = {str(h["context_id"]): h for h in row["hits"]}
            local_ce = {}
            for c in pool_ctx:
                value = ce_value.get((qid, c))
                if value is None:
                    raise SystemExit(f"[apply-ranking-rules] thiếu logit CE cho qid={qid} context={c}")
                local_ce[c] = value
            rank_ce = _rank_ce_with_tiebreak(pool_ctx, local_ce, fusion_rank)
            base[qid] = _blend_full_order(pool_ctx, fusion_rank, rank_ce, BLEND_WEIGHT, BLEND_RRF_K)
            pool[qid] = [
                {
                    "query_id": qid,
                    "context_id": c,
                    "fusion_rank": fusion_rank[c],
                    "bm25_rank": hits_by_ctx.get(c, {}).get("bm25_rank"),
                    "vae_rank": hits_by_ctx.get(c, {}).get("vae_rank"),
                    "bgem3_rank": hits_by_ctx.get(c, {}).get("bgem3_rank"),
                    "f2llm_rank": hits_by_ctx.get(c, {}).get("f2llm_rank"),
                    "ce_value": local_ce[c],
                }
                for c in pool_ctx
            ]
    return pool, base


def _apply_rule_chain(
    pool: dict[str, list[dict[str, Any]]],
    base: dict[str, list[str]],
    qids: list[str],
    n_train: dict[str, int],
    questions: dict[str, str],
) -> dict[str, dict[str, list[str]]]:
    """Giai đoạn 1 → 2 → 3 → luật năm, theo tham số cố định."""
    p1 = STAGE1_PARAMS
    p2 = {"a": 0.0075, "w": 0.45, "k": 10.0, "knn_k": 1, "knn_tau": 0.45, "knn_beta": 0.16, "cogold_gamma": 0.01}
    stage1 = _apply_stage1_stage2(pool, qids, n_train, [], a=p1["a"], w=p1["w"], k=p1["k"], margin=p1["margin"])
    ctx = _build_question_context(qids)
    cog = {q: _cooccurrence_bonus(ctx, stage1[q][:COOCCURRENCE_ANCHOR_TOP]) for q in qids}
    knn = {q: _neighbor_question_bonus(ctx, q, p2["knn_k"], p2["knn_tau"]) for q in qids}
    stage2 = _apply_stage1_stage2(
        pool, qids, n_train, [(p2["knn_beta"], knn), (p2["cogold_gamma"], cog)], a=p2["a"], w=p2["w"], k=p2["k"], margin=p1["margin"]
    )
    stage3, _audit = _stage3_apply("combined", stage2, base, pool, n_train, questions)
    question_years = {q: set(YEAR_RULE_PATTERN.findall(questions.get(q, ""))) for q in qids}
    question_years = {q: years for q, years in question_years.items() if years}
    needed = {c for q in question_years for c in stage3[q][:YEAR_RULE_WINDOW]}
    final, _touched = _year_rule_apply(stage3, question_years, _year_rule_context_years(needed), YEAR_RULE_WINDOW)
    return {"stage1": stage1, "stage2": stage2, "stage3": stage3, "final": final}


def cmd_apply_ranking_rules(args: argparse.Namespace) -> int:
    """Áp tham số cố định lên fused + logit CE của một tập câu hỏi bất kỳ."""
    global TRAIN_LABELS_JSON, QUESTIONS_JSON, CHUNKS_JSONL
    TRAIN_LABELS_JSON, QUESTIONS_JSON = Path(args.train_json), Path(args.questions)
    CHUNKS_JSONL = Path(args.corpus)
    out_dir = Path(args.out_dir)

    pool, base = _build_candidate_pool(Path(args.fused), Path(args.ce_logits))

    # n_train(d) = số câu train có gold d, đếm thẳng từ nhãn train.
    n_train: dict[str, int] = {}
    for gold in _load_train_labels()[2]:
        for doc in gold:
            n_train[doc] = n_train.get(doc, 0) + 1
    raw_questions = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    questions = {str(qid): str(item["question"]) for qid, item in raw_questions.items()}
    qids = list(questions)
    no_pool = [qid for qid in qids if qid not in pool]
    if no_pool:
        raise SystemExit(f"[apply-ranking-rules] {len(no_pool)} qid trong file câu hỏi không có pool fusion/CE, ví dụ {no_pool[:5]}")
    pool = {qid: pool[qid] for qid in qids}
    base = {qid: base[qid] for qid in qids}
    chain = _apply_rule_chain(pool, base, qids, n_train, questions)

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "top200.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for qid in qids:
            for rank, context_id in enumerate(chain["final"][qid], 1):
                stream.write(json.dumps({"query_id": qid, "context_id": context_id, "rank": rank}, ensure_ascii=False) + "\n")
    submission = {qid: {"answer": chain["final"][qid][:5]} for qid in qids}
    (out_dir / "submission.json").write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    with zipfile.ZipFile(out_dir / "submission.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(out_dir / "submission.json", arcname="submission.json")

    report: dict[str, Any] = {"n_queries": len(qids)}
    report["output_top200_sha256"] = _hash_file(out_dir / "top200.jsonl")
    report["output_submission_sha256"] = _hash_file(out_dir / "submission.json")
    report["output_submission_zip_sha256"] = _hash_file(out_dir / "submission.zip")
    if args.reference_ranking:
        # So từng câu (đủ 200 vị trí) với một ranking đã lưu.
        reference: dict[str, list[tuple[int, str]]] = {}
        with Path(args.reference_ranking).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    row = json.loads(line)
                    reference.setdefault(str(row["query_id"]), []).append((int(row["rank"]), str(row["context_id"])))
        mismatched = [q for q, rows in reference.items() if [c for _, c in sorted(rows)] != chain["final"].get(q)]
        report["reference_check"] = {
            "path": str(args.reference_ranking), "n_reference_queries": len(reference),
            "n_mismatched": len(mismatched), "mismatched_qids_head": mismatched[:10],
        }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report.get("reference_check", {}).get("n_mismatched") else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Áp tham số cố định lên fused-ranking + logit cross-encoder.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    apply_cmd = sub.add_parser(
        "apply-ranking-rules",
        help="áp tham số cố định lên fused + logit CE của một tập câu hỏi bất kỳ",
    )
    apply_cmd.add_argument("--fused", required=True, help="ranked.jsonl của fusion nhiều nguồn")
    apply_cmd.add_argument("--ce-logits", required=True, help="logit CE theo context: {query_id, context_id, value}")
    apply_cmd.add_argument("--questions", required=True, help="câu hỏi cần xếp hạng: {qid: {question}}")
    apply_cmd.add_argument("--train-json", required=True, help="nhãn train: {qid: {question, answer}}")
    apply_cmd.add_argument("--corpus", required=True, help="retrieval_chunks.jsonl")
    apply_cmd.add_argument("--out-dir", required=True)
    apply_cmd.add_argument("--reference-ranking", default=None, help="tuỳ chọn: ranking top-200 đã lưu để so từng câu")
    apply_cmd.set_defaults(fn=cmd_apply_ranking_rules)
    args = parser.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
