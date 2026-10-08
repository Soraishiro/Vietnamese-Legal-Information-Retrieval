"""Xếp lại ứng viên bằng cross-encoder và kiểm tra ngân sách tham số của hệ thống.

File này làm gì
----------------
Cho mỗi câu hỏi và mỗi ứng viên, chấm điểm bằng cross-encoder đã tinh chỉnh, rồi ghi logit
theo từng context để bước sau dùng. Trước khi chạy, file kiểm tra tổng tham số toàn hệ
thống có vượt trần 4 tỷ hay không, dựa trên `system_inventory.json` và hash trong
`manifest.json` của checkpoint.

Pipeline stage
--------------
Sau fusion, trước bước xếp hạng cuối.

Input
-----
  --queries      file câu hỏi cần xếp: {qid: {"question": ...}}
  --fused        ranked.jsonl của fusion (lấy danh sách ứng viên)
  --out-dir      thư mục ghi logit
  --model-name   thư mục checkpoint cross-encoder
  --inventory    file system_inventory.json

Output
-------
  <out-dir>/signals/ce_context_max.jsonl   logit theo (query_id, context_id)
  <out-dir>/report.json                    ngân sách tham số, sha256, số lượng

Model/class
-----------
  transformers.AutoModelForSequenceClassification — bge-reranker-v2-m3 đã tinh chỉnh,
  checkpoint trong `checkpoints/ce_champion_epoch2/`. Chỉ chạy trên CUDA.

Command
-------
  python rerank.py rerank --queries <Q> --fused <fused.jsonl> --out-dir <R>/rerank_private \
      --model-name checkpoints/ce_champion_epoch2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence

from legal_ir.core import commons
from legal_ir.core.commons import artifact_provenance as _artifact_provenance, canonical_json_sha256 as semantic_sha256, canonical_json_value as _semantic_value, VIETNAMESE_QUERY_STOPWORDS, checkpoint_parameter_count, expand_legal_abbreviations, infer_view, iter_jsonl, load_queries, load_ranked_jsonl as load_ranked, normalize_inline, normalize_query, word_tokens, read_json_mapping as _read_json_object, read_json_mapping as _read_json_mapping, sha256_file, sha256_file as _hash_file, snapshot_file_inventory as snapshot_model_runtime_inventory, verify_file_inventory as verify_model_runtime_closure
def _trace(event: str, **fields: Any) -> None:
    payload = {
        "event": event,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "pid": os.getpid(),
        **fields,
    }
    print(
        "[RERANK-TRACE] "
        + json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str),
        flush=True,
    )


def source_closure_sha256() -> tuple[str, dict[str, str]]:
    paths = {
        "rerank": Path(__file__).resolve(),
        "commons": Path(str(commons.__file__)).resolve(),
    }
    hashes: dict[str, str] = {}
    for key, candidate in paths.items():
        if not candidate.is_file():
            raise FileNotFoundError(f"source closure file is missing: {key}={candidate}")
        hashes[key] = _hash_file(candidate)
    return semantic_sha256(hashes), hashes

def load_verified_judge_model(
    manifest_path: str | Path,
) -> tuple[JudgeDescriptor, Path, dict[str, Any]]:
    path = Path(manifest_path)
    payload = _read_json_object(path)
    schema = payload.get("schema_version")
    if schema == "rerank-nmodels-champion-runtime-manifest-v1":
        return _load_runtime_manifest_closure(path, payload)
    if schema not in ("judge-model-manifest-v1", "rerank-judge-manifest-v1"):
        raise ValueError(f"Unsupported Judge model manifest: {path}")
    raw_dir = payload.get("model_dir", payload.get("model_name"))
    if not isinstance(raw_dir, str) or not raw_dir:
        raise ValueError("Judge model manifest lacks a model directory")
    model_dir = Path(raw_dir)
    for field in (
        "adapter_id",
        "adapter_version",
        "model_class",
        "output_kinds",
        "eligibility",
    ):
        if field not in payload:
            raise ValueError(f"Judge model manifest lacks {field}")
    declared_files = {
        "model.safetensors": payload.get("checkpoint_sha256"),
        "tokenizer_config.json": payload.get("tokenizer_sha256"),
        "config.json": payload.get("config_sha256"),
    }
    for filename, expected in declared_files.items():
        candidate = model_dir / filename
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        if not isinstance(expected, str) or _hash_file(candidate) != expected:
            raise ValueError(f"Judge model {filename} differs from manifest")
    actual_count, checkpoint_files = checkpoint_parameter_count(model_dir)
    inventory = snapshot_model_runtime_inventory(model_dir)
    config_path = model_dir / "config.json"
    descriptor = JudgeDescriptor(
        adapter_id=str(payload["adapter_id"]),
        adapter_version=str(payload["adapter_version"]),
        model_class=str(payload["model_class"]),
        checkpoint_sha256=semantic_sha256(checkpoint_files),
        tokenizer_sha256=semantic_sha256(inventory),
        config_sha256=_hash_file(config_path) if config_path.is_file() else "",
        output_kinds=tuple(str(item) for item in payload["output_kinds"]),
        eligibility=str(payload["eligibility"]),
        parameter_count=actual_count,
    )
    report: dict[str, Any] = {
        "manifest_schema": schema,
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _hash_file(path),
        "model_dir": str(model_dir.resolve()),
        "inventory_pinned": False,
        "runtime_file_inventory_hash": semantic_sha256(inventory),
        "runtime_file_count": len(inventory),
        "declared_parameter_count": payload.get("parameter_count"),
        "verified_parameter_count": actual_count,
        "checkpoint_files": checkpoint_files,
    }
    return descriptor, model_dir.resolve(), report

def _load_runtime_manifest_closure(
    path: Path, payload: dict[str, Any]
) -> tuple[JudgeDescriptor, Path, dict[str, Any]]:
    claimed_hash = payload.pop("content_hash", None)
    unsigned = dict(payload)
    if claimed_hash != semantic_sha256(unsigned):
        raise ValueError("Champion runtime manifest content hash mismatch")
    model_dir_value = payload.get("model_dir")
    if not isinstance(model_dir_value, str):
        raise ValueError("Champion runtime manifest lacks a model directory")
    pinned_inventory = payload.get("runtime_files")
    if not isinstance(pinned_inventory, dict) or not pinned_inventory:
        raise ValueError("Champion runtime manifest lacks a file inventory")
    live_inventory = verify_model_runtime_closure(model_dir_value, pinned_inventory)
    stored = payload.get("descriptor")
    if not isinstance(stored, Mapping):
        raise ValueError("Champion runtime manifest lacks a descriptor")
    checkpoint_count, checkpoint_files = checkpoint_parameter_count(model_dir_value)
    live_config = Path(model_dir_value) / "config.json"
    descriptor = JudgeDescriptor(
        adapter_id=str(stored["adapter_id"]),
        adapter_version=str(stored["adapter_version"]),
        model_class=str(stored["model_class"]),
        checkpoint_sha256=semantic_sha256(checkpoint_files),
        tokenizer_sha256=semantic_sha256(live_inventory),
        config_sha256=_hash_file(live_config) if live_config.is_file() else "",
        output_kinds=tuple(str(item) for item in stored["output_kinds"]),
        eligibility=str(stored["eligibility"]),
        parameter_count=checkpoint_count,
    )
    if dict(payload["descriptor"]) != {
        **asdict(descriptor),
        "output_kinds": list(descriptor.output_kinds),
    }:
        raise ValueError("Champion runtime manifest descriptor differs")
    report: dict[str, Any] = {
        "manifest_schema": "rerank-nmodels-champion-runtime-manifest-v1",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": claimed_hash,
        "model_dir": str(Path(model_dir_value).resolve()),
        "inventory_pinned": True,
        "runtime_file_inventory_hash": semantic_sha256(live_inventory),
        "runtime_file_count": len(live_inventory),
        "declared_parameter_count": stored.get("parameter_count"),
        "verified_parameter_count": checkpoint_count,
        "checkpoint_files": checkpoint_files,
    }
    return descriptor, Path(model_dir_value).resolve(), report

@dataclass(frozen=True)
class Candidate:
    query_id: str
    context_id: str
    universe_position: int

@dataclass(frozen=True)
class CandidateUniverse:
    query_order: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    content_hash: str

    @classmethod
    def from_rankings(
        cls,
        query_order: Sequence[str],
        ranked_by_query: Mapping[str, Sequence[str]],
    ) -> "CandidateUniverse":
        order = tuple(str(query_id) for query_id in query_order)
        if len(set(order)) != len(order):
            raise ValueError("duplicate query_id in Candidate Universe")
        candidates: list[Candidate] = []
        for query_id in order:
            if query_id not in ranked_by_query:
                raise ValueError(f"missing ranking for query_id={query_id}")
            seen: set[str] = set()
            for position, raw_context_id in enumerate(ranked_by_query[query_id], 1):
                context_id = str(raw_context_id)
                if not context_id:
                    raise ValueError(f"empty candidate query_id={query_id}")
                if context_id in seen:
                    raise ValueError(
                        f"duplicate candidate query_id={query_id} context_id={context_id}"
                    )
                seen.add(context_id)
                candidates.append(Candidate(query_id, context_id, position))
        rows = tuple(candidates)
        return cls(
            query_order=order,
            candidates=rows,
            content_hash=semantic_sha256((order, rows)),
        )

    def context_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(
            row.context_id for row in self.candidates if row.query_id == query_id
        )

@dataclass(frozen=True)
class SignalProvenance:
    producer_stage: str
    producer_type: str
    source: str
    config_hash: str
    adapter_id: str
    adapter_version: str
    model_manifest_sha256: str
    input_artifact_hashes: tuple[tuple[str, str], ...]
    produced_at_utc: str

@dataclass(frozen=True)
class RankSignal:
    query_id: str
    context_id: str
    rank: int
    channel: str
    view_id: str
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not self.channel:
            raise ValueError("rank signal requires channel")

@dataclass(frozen=True)
class ScoreSignal:

    query_id: str
    context_id: str
    value: float
    channel: str
    view_id: str
    scale: str
    orientation: str
    entity_scope: str
    provenance: SignalProvenance
    chunk_id: str | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("score signal value must be finite")
        if self.orientation not in {"higher_is_better", "lower_is_better"}:
            raise ValueError("unsupported score orientation")
        if self.entity_scope not in {"chunk", "context"}:
            raise ValueError("score signal entity_scope must be chunk or context")
        if self.entity_scope == "chunk" and not self.chunk_id:
            raise ValueError("chunk score signal requires chunk_id")
        if not self.channel or not self.scale:
            raise ValueError("score signal requires channel and scale")

@dataclass(frozen=True)
class PreferenceSignal:

    query_id: str
    left_context_id: str
    right_context_id: str
    outcome: str
    confidence: float | None
    channel: str
    view_id: str
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        if self.outcome not in {"left", "right", "tie"}:
            raise ValueError("preference outcome must be left, right or tie")
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError("preference confidence must be finite")

@dataclass(frozen=True)
class JudgementSignal:

    query_id: str
    context_id: str
    label: str
    confidence: float | None
    rationale_ref: str | None
    parser_schema_version: str
    channel: str
    view_id: str
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        if not self.label or not self.parser_schema_version or not self.channel:
            raise ValueError("judgement signal requires label, schema version and channel")
        if self.confidence is not None and not math.isfinite(self.confidence):
            raise ValueError("judgement confidence must be finite")

@dataclass(frozen=True)
class FeatureSignal:

    query_id: str
    context_id: str
    feature_name: str
    feature_version: str
    value: float | None
    channel: str
    view_id: str
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        if not self.feature_name or not self.feature_version or not self.channel:
            raise ValueError("feature signal requires name, version and channel")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("feature value must be finite")

@dataclass(frozen=True)
class SelectionSignal:

    query_id: str
    context_id: str
    selected: bool
    channel: str
    view_id: str
    reason: str
    provenance: SignalProvenance

    def __post_init__(self) -> None:
        if not self.channel or not self.view_id or not self.reason:
            raise ValueError("selection signal requires channel, view and reason")

RankingSignal = (
    ScoreSignal
    | RankSignal
    | PreferenceSignal
    | JudgementSignal
    | FeatureSignal
    | SelectionSignal
)

@dataclass(frozen=True)
class SignalLedgerSnapshot:
    signals: tuple[RankingSignal, ...]
    content_hash: str

class TypedSignalLedger:

    def __init__(self, universe: CandidateUniverse) -> None:
        self._universe = universe
        self._signals: tuple[RankingSignal, ...] = ()

    def append_batch(
        self, signals: Sequence[RankingSignal]
    ) -> SignalLedgerSnapshot:
        batch = tuple(signals)
        allowed = {
            (row.query_id, row.context_id) for row in self._universe.candidates
        }
        for signal in batch:
            references = (
                (
                    (signal.query_id, signal.left_context_id),
                    (signal.query_id, signal.right_context_id),
                )
                if isinstance(signal, PreferenceSignal)
                else ((signal.query_id, signal.context_id),)
            )
            if any(reference not in allowed for reference in references):
                raise ValueError("signal references candidate outside Candidate Universe")
        self._signals = self._signals + batch
        return self.snapshot()

    def snapshot(self) -> SignalLedgerSnapshot:
        return SignalLedgerSnapshot(
            signals=self._signals,
            content_hash=semantic_sha256(self._signals),
        )

def ranking_source_name(path: str | Path) -> str:
    candidate = Path(path)
    stem = candidate.stem
    if stem != "ranked":
        return stem
    parent = candidate.parent.name
    return parent.split("_", 1)[0]

def load_candidate_inputs(
    args: Any,
) -> tuple[
    Mapping[str, str],
    CandidateUniverse,
    tuple[RankingSignal, ...],
    Mapping[tuple[str, str], Mapping[str, Any]],
    tuple[Mapping[tuple[str, str], Mapping[str, Any]], ...],
]:
    _trace(
        "inputs.start",
        queries=str(Path(args.queries).resolve()),
        corpus=str(Path(args.corpus).resolve()),
        fused_ranked=str(Path(args.fused_ranked).resolve()),
        candidate_depth=args.candidate_depth,
    )
    query_rows = load_queries(args.queries)
    questions = {str(row["query_id"]): str(row["question"]) for row in query_rows}
    order, fused = load_ranked(args.fused_ranked)
    if tuple(order) != tuple(questions):
        raise ValueError("Query order differs between queries and fused ranking")
    ranked_by_query = {
        query_id: tuple(fused[query_id]["ranked"][: args.candidate_depth])
        for query_id in order
    }
    if any(not rows for rows in ranked_by_query.values()):
        raise ValueError("Candidate depth produced an empty query ranking")
    universe = CandidateUniverse.from_rankings(order, ranked_by_query)
    fusion_provenance = SignalProvenance(
        producer_stage="fusion",
        producer_type="input_ranking",
        source=str(Path(args.fused_ranked).resolve()),
        config_hash=sha256_file(args.fused_ranked),
        adapter_id="input_ranking",
        adapter_version="v1",
        model_manifest_sha256="not_applicable",
        input_artifact_hashes=(("fused_ranked", sha256_file(args.fused_ranked)),),
        produced_at_utc=datetime.now(UTC).isoformat(),
    )
    initial_signals: list[RankingSignal] = [
        RankSignal(
            query_id=query_id,
            context_id=context_id,
            rank=rank,
            channel="fusion_rank",
            view_id="universe",
            provenance=fusion_provenance,
        )
        for query_id in order
        for rank, context_id in enumerate(ranked_by_query[query_id], 1)
    ]
    for query_id in order:
        for context_id in ranked_by_query[query_id]:
            fused_value = fused[query_id]["evidence"].get(context_id, {}).get("score")
            if isinstance(fused_value, (int, float)):
                initial_signals.append(
                    ScoreSignal(
                        query_id=query_id,
                        context_id=context_id,
                        value=float(fused_value),
                        channel="fusion_score",
                        view_id="universe",
                        scale="weighted_rrf",
                        orientation="higher_is_better",
                        entity_scope="context",
                        provenance=fusion_provenance,
                    )
                )


    fused_hits: dict[tuple[str, str], Mapping[str, Any]] = {}
    for query_id in order:
        for fusion_rank, context_id in enumerate(ranked_by_query[query_id], 1):
            hit = dict(fused[query_id]["evidence"].get(context_id, {}))
            hit["fusion_rank"] = fusion_rank
            fused_hits[(query_id, context_id)] = hit
    branches: list[Mapping[tuple[str, str], Mapping[str, Any]]] = []
    source_names: set[str] = {"fusion"}
    for path in [args.bm25_ranked, *args.dense_ranked]:
        if path is None:
            continue
        branch_order, branch = load_ranked(path)
        if tuple(branch_order) != tuple(order):
            raise ValueError(f"Query order differs in branch ranking: {path}")
        source_name = ranking_source_name(path)
        if source_name in source_names:
            raise ValueError(f"Duplicate ranking source name={source_name}: {path}")
        source_names.add(source_name)
        provenance = SignalProvenance(
            producer_stage="retrieval",
            producer_type="input_ranking",
            source=str(Path(path).resolve()),
            config_hash=sha256_file(path),
            adapter_id="input_ranking",
            adapter_version="v1",
            model_manifest_sha256="not_applicable",
            input_artifact_hashes=(("ranking", sha256_file(path)),),
            produced_at_utc=datetime.now(UTC).isoformat(),
        )
        candidate_pairs = {
            (query_id, context_id)
            for query_id in order
            for context_id in ranked_by_query[query_id]
        }
        for query_id in order:
            for rank, context_id in enumerate(branch[query_id]["ranked"], 1):
                if (query_id, context_id) not in candidate_pairs:
                    continue
                initial_signals.append(
                    RankSignal(
                        query_id=query_id,
                        context_id=context_id,
                        rank=rank,
                        channel=f"{source_name}_rank",
                        view_id="universe",
                        provenance=provenance,
                    )
                )
                raw_value = branch[query_id]["evidence"].get(context_id, {}).get("raw_score")
                if isinstance(raw_value, (int, float)):
                    initial_signals.append(
                        ScoreSignal(
                            query_id=query_id,
                            context_id=context_id,
                            value=float(raw_value),
                            channel=f"{source_name}_raw_score",
                            view_id="universe",
                            scale=f"{source_name}_raw_score",
                            orientation="higher_is_better",
                            entity_scope="context",
                            provenance=provenance,
                        )
                    )
        branch_evidence: dict[tuple[str, str], Mapping[str, Any]] = {}
        for query_id in order:
            rank_by_context = {
                context_id: rank
                for rank, context_id in enumerate(branch[query_id]["ranked"], 1)
            }
            for context_id in ranked_by_query[query_id]:
                hit = dict(branch[query_id]["evidence"].get(context_id, {}))
                chunk_id = hit.pop("chunk_id", None)
                if isinstance(chunk_id, str) and chunk_id:
                    hit[f"{source_name}_chunk_id"] = chunk_id
                    hit[f"{source_name}_rank"] = rank_by_context.get(context_id)
                    raw_score = hit.get("raw_score")
                    hit[f"{source_name}_raw_score"] = (
                        float(raw_score)
                        if isinstance(raw_score, (int, float))
                        else None
                    )
                branch_evidence[(query_id, context_id)] = hit
        branches.append(branch_evidence)
    _trace(
        "inputs.complete",
        queries=len(questions),
        candidates=len(universe.candidates),
        initial_signals=len(initial_signals),
        branch_sources=len(branches),
    )
    return questions, universe, tuple(initial_signals), fused_hits, tuple(branches)


PairKey = tuple[str, str]

_CHUNK_ID_RE = re.compile(r"^ctx(\d+)__([a-z]+)(\d+)$")

def _chunk_neighbors(chunk_id: str, steps: int) -> tuple[list[str], list[str]]:
    match = _CHUNK_ID_RE.match(chunk_id)
    if not match:
        return [], []
    context_id, prefix, index = match.group(1), match.group(2), match.group(3)
    width, value = len(index), int(index)
    previous = [
        f"ctx{context_id}__{prefix}{value - offset:0{width}d}"
        for offset in range(1, steps + 1)
        if value - offset >= 0
    ]
    following = [
        f"ctx{context_id}__{prefix}{value + offset:0{width}d}"
        for offset in range(1, steps + 1)
    ]
    return previous, following

def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)

def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)

@dataclass(frozen=True)
class EvidenceAttribution:

    source: str | None
    source_rank: int | None
    source_raw_score: float | None

@dataclass(frozen=True)
class EvidenceRecord:

    query_id: str
    context_id: str
    chunk_id: str
    view: str
    text: str
    text_sha256: str
    attributions: tuple[EvidenceAttribution, ...]
    resolution_reason: str
    corpus_sha256: str

@dataclass(frozen=True)
class EvidenceBatch:

    records: tuple[EvidenceRecord, ...]
    content_hash: str

    def for_pair(self, query_id: str, context_id: str) -> tuple[EvidenceRecord, ...]:
        return tuple(
            row
            for row in self.records
            if row.query_id == query_id and row.context_id == context_id
        )

def order_pointwise_evidence_by_length(
    evidence: EvidenceBatch, questions: Mapping[str, str]
) -> EvidenceBatch:
    records = tuple(
        sorted(
            evidence.records,
            key=lambda row: len(questions[row.query_id]) + len(row.text),
        )
    )
    return EvidenceBatch(records=records, content_hash=semantic_sha256(records))

@dataclass(frozen=True)
class EvidencePolicy:

    mode: str
    max_chunks: int
    use_article_view: bool
    include_metadata: bool
    neighbor_context: int = 0
    source_priority: tuple[str, ...] = ()


    question_order_min_gain: int = 0

    def __post_init__(self) -> None:
        if self.mode not in {"champion_v1", "canonical_evidence"}:
            raise ValueError(f"Unsupported evidence mode: {self.mode}")
        if self.max_chunks < 1:
            raise ValueError("max_chunks must be positive")
        if self.neighbor_context < 0:
            raise ValueError("neighbor_context must be non-negative")
        if self.question_order_min_gain < 0:
            raise ValueError("question_order_min_gain must be non-negative")
        if any(not source for source in self.source_priority):
            raise ValueError("source_priority entries must be non-empty")
        if len(set(self.source_priority)) != len(self.source_priority):
            raise ValueError("source_priority entries must be unique")
        if self.mode == "canonical_evidence" and not self.source_priority:
            raise ValueError("canonical_evidence requires explicit source_priority")
        if self.mode == "champion_v1" and self.source_priority:
            raise ValueError("champion_v1 does not use canonical source_priority")

    @classmethod
    def champion_v1(
        cls,
        max_chunks: int,
        use_article_view: bool = True,
        neighbor_context: int = 0,
        question_order_min_gain: int = 0,
    ) -> "EvidencePolicy":
        return cls(
            mode="champion_v1",
            max_chunks=max_chunks,
            use_article_view=use_article_view,
            include_metadata=True,
            neighbor_context=neighbor_context,
            question_order_min_gain=question_order_min_gain,
        )

class CorpusChangedError(RuntimeError):
    pass

@dataclass(frozen=True)
class CanonicalChunk:
    context_id: str
    chunk_id: str
    view: str
    text: str

@dataclass(frozen=True)
class _ChunkOffset:
    context_id: str
    chunk_id: str
    view: str
    offset: int
    length: int
    corpus_order: int

@dataclass(frozen=True)
class _StoreMaterial:
    text_by_chunk: Mapping[str, CanonicalChunk]
    fallback_by_context: Mapping[str, Mapping[str, CanonicalChunk]]
    missing_requested_chunk_ids: tuple[str, ...]

@dataclass(frozen=True)
class AvailableEvidence:

    query_id: str
    context_id: str
    chunk_id: str
    attributions: tuple[EvidenceAttribution, ...]
    discovery_order: int

@dataclass(frozen=True)
class EvidenceAvailability:

    records: tuple[AvailableEvidence, ...]
    content_hash: str
    _records_by_pair: Mapping[PairKey, tuple[AvailableEvidence, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        grouped: dict[PairKey, list[AvailableEvidence]] = defaultdict(list)
        for record in self.records:
            grouped[(record.query_id, record.context_id)].append(record)
        object.__setattr__(
            self,
            "_records_by_pair",
            MappingProxyType(
                {pair: tuple(values) for pair, values in grouped.items()}
            ),
        )

    def for_pair(
        self, query_id: str, context_id: str
    ) -> tuple[AvailableEvidence, ...]:
        return self._records_by_pair.get((query_id, context_id), ())

    def filter_pairs(self, pairs: Sequence[PairKey]) -> "EvidenceAvailability":
        records = tuple(
            record
            for pair in pairs
            for record in self._records_by_pair.get(pair, ())
        )
        return EvidenceAvailability(
            records=records,
            content_hash=semantic_sha256(records),
        )

class EvidenceSelector:

    @staticmethod
    def select(
        universe: CandidateUniverse,
        availability: EvidenceAvailability,
        policy: EvidencePolicy,
    ) -> dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]]:
        selected: dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]] = {}
        priority = {
            source: index for index, source in enumerate(policy.source_priority)
        }
        for candidate in universe.candidates:
            pair = (candidate.query_id, candidate.context_id)
            rows = list(availability.for_pair(*pair))
            if policy.mode == "canonical_evidence":
                named_sources = {
                    attribution.source
                    for row in rows
                    for attribution in row.attributions
                    if attribution.source is not None
                }
                missing_sources = sorted(named_sources - set(priority))
                if missing_sources:
                    raise ValueError(
                        "canonical source_priority lacks named sources: "
                        f"{missing_sources}"
                    )

                def attribution_key(
                    attribution: EvidenceAttribution,
                ) -> tuple[float, int, str]:
                    if attribution.source is None:
                        return (math.inf, len(priority), "")
                    return (
                        float(attribution.source_rank)
                        if attribution.source_rank is not None
                        else math.inf,
                        priority[attribution.source],
                        attribution.source,
                    )

                rows.sort(
                    key=lambda row: (
                        min(attribution_key(item) for item in row.attributions),
                        row.chunk_id,
                    )
                )
                selected[pair] = [
                    (
                        row.chunk_id,
                        tuple(sorted(row.attributions, key=attribution_key)),
                    )
                    for row in rows
                ]
            else:
                rows.sort(key=lambda row: row.discovery_order)
                selected[pair] = [
                    (row.chunk_id, row.attributions) for row in rows
                ]
        return selected

class CanonicalChunkStore:

    def __init__(self, corpus: str | Path) -> None:
        self.path = Path(corpus)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        _trace("corpus_index.start", path=str(self.path.resolve()), size_bytes=self.path.stat().st_size)
        self._offset_by_chunk: dict[str, _ChunkOffset] = {}
        offsets_by_context: dict[str, list[_ChunkOffset]] = defaultdict(list)
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            before = self._identity(os.fstat(handle.fileno()))
            for corpus_order, raw_line in enumerate(handle):
                if corpus_order and corpus_order % 100_000 == 0:
                    _trace(
                        "corpus_index.progress",
                        path=str(self.path.resolve()),
                        rows=corpus_order,
                        chunks=len(self._offset_by_chunk),
                    )
                offset = handle.tell() - len(raw_line)
                digest.update(raw_line)
                if not raw_line.strip():
                    continue
                row = self._parse_row(raw_line, corpus_order + 1)
                chunk_id = row["chunk_id"]
                if chunk_id in self._offset_by_chunk:
                    raise ValueError(
                        f"Duplicate chunk_id at row {corpus_order + 1}: {chunk_id}"
                    )
                reference = _ChunkOffset(
                    context_id=row["context_id"],
                    chunk_id=chunk_id,
                    view=infer_view(chunk_id),
                    offset=offset,
                    length=len(raw_line),
                    corpus_order=corpus_order,
                )
                self._offset_by_chunk[chunk_id] = reference
                offsets_by_context[reference.context_id].append(reference)
            after = self._identity(os.fstat(handle.fileno()))
        if before != after:
            raise CorpusChangedError(
                f"corpus changed while building byte-offset index: {self.path}"
            )
        if self._identity(self.path.stat()) != after:
            raise CorpusChangedError(
                f"corpus path changed while building byte-offset index: {self.path}"
            )
        self._file_identity = after
        self._offsets_by_context = {
            context_id: tuple(references)
            for context_id, references in offsets_by_context.items()
        }
        self.corpus_sha256 = digest.hexdigest()
        _trace(
            "corpus_index.complete",
            path=str(self.path.resolve()),
            rows=sum(len(rows) for rows in self._offsets_by_context.values()),
            chunks=len(self._offset_by_chunk),
            contexts=len(self._offsets_by_context),
            corpus_sha256=self.corpus_sha256,
        )

    @staticmethod
    def _identity(stat: os.stat_result) -> tuple[int, int, int, int]:
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    @staticmethod
    def _parse_row(raw_line: bytes, line_number: int) -> dict[str, str]:
        try:
            text = raw_line.decode("utf-8")
            if line_number == 1:
                text = text.removeprefix("\ufeff")
            row = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid UTF-8 JSONL at corpus row {line_number}"
            ) from exc
        if not isinstance(row, Mapping) or set(row) != {"chunk_id", "context_id", "text"}:
            raise ValueError(
                f"Retrieval schema mismatch at corpus row {line_number}"
            )
        chunk_id = normalize_inline(row["chunk_id"])
        context_id = normalize_inline(row["context_id"])
        normalized_text = normalize_query(row["text"])
        if not chunk_id or not context_id or not normalized_text:
            raise ValueError(f"Empty retrieval field at corpus row {line_number}")
        if not chunk_id.startswith(f"ctx{context_id}__"):
            raise ValueError(
                f"chunk_id/context_id mismatch at corpus row {line_number}: "
                f"{chunk_id!r} vs {context_id!r}"
            )
        return {
            "chunk_id": chunk_id,
            "context_id": context_id,
            "text": normalized_text,
        }

    def _verify_open_file(self, handle: Any) -> None:
        path_identity = self._identity(self.path.stat())
        open_identity = self._identity(os.fstat(handle.fileno()))
        if path_identity != self._file_identity or open_identity != self._file_identity:
            raise CorpusChangedError(
                f"corpus changed after byte-offset index build: {self.path}"
            )

    def verify_current(self) -> None:
        with self.path.open("rb") as handle:
            self._verify_open_file(handle)

    def _read_references(
        self, references: Sequence[_ChunkOffset]
    ) -> tuple[CanonicalChunk, ...]:
        if not references:
            return ()
        chunks: list[CanonicalChunk] = []
        with self.path.open("rb") as handle:
            self._verify_open_file(handle)
            for reference in references:
                handle.seek(reference.offset)
                raw_line = handle.read(reference.length)
                row = self._parse_row(raw_line, reference.corpus_order + 1)
                if (
                    row["chunk_id"] != reference.chunk_id
                    or row["context_id"] != reference.context_id
                ):
                    raise CorpusChangedError(
                        "corpus changed: indexed row identity no longer matches "
                        f"chunk_id={reference.chunk_id}"
                    )
                chunks.append(
                    CanonicalChunk(
                        context_id=reference.context_id,
                        chunk_id=reference.chunk_id,
                        view=reference.view,
                        text=row["text"],
                    )
                )
            self._verify_open_file(handle)
        return tuple(chunks)

    def get_chunk(self, chunk_id: str) -> CanonicalChunk:
        reference = self._offset_by_chunk.get(str(chunk_id))
        if reference is None:
            raise KeyError(f"Unknown canonical chunk_id={chunk_id}")
        return self._read_references((reference,))[0]

    def context_id_for_chunk(self, chunk_id: str) -> str:
        reference = self._offset_by_chunk.get(str(chunk_id))
        if reference is None:
            raise KeyError(f"Unknown canonical chunk_id={chunk_id}")
        return reference.context_id

    def get_context_chunks(
        self,
        context_id: str,
        views: Sequence[str] | None = None,
    ) -> tuple[CanonicalChunk, ...]:
        references = self._offsets_by_context.get(str(context_id), ())
        if views is not None:
            allowed_views = frozenset(str(view) for view in views)
            references = tuple(
                reference
                for reference in references
                if reference.view in allowed_views
            )
        return self._read_references(references)

    def materialize(
        self,
        requested_chunk_ids: set[str],
        candidate_context_ids: set[str],
        policy: EvidencePolicy,
    ) -> _StoreMaterial:
        neighbor_ids: set[str] = set()
        if policy.neighbor_context:
            for chunk_id in requested_chunk_ids:
                previous, following = _chunk_neighbors(chunk_id, policy.neighbor_context)
                neighbor_ids.update(previous)
                neighbor_ids.update(following)
        keep_chunks = requested_chunk_ids | neighbor_ids
        allowed_views = {"packed", "recovery", "short"}
        if policy.include_metadata:
            allowed_views.add("metadata")
        if policy.use_article_view:
            allowed_views.add("article")
        references_by_chunk: dict[str, _ChunkOffset] = {
            chunk_id: reference
            for chunk_id in keep_chunks
            if (reference := self._offset_by_chunk.get(chunk_id)) is not None
        }
        fallback_references: dict[str, dict[str, _ChunkOffset]] = defaultdict(dict)
        for context_id in candidate_context_ids:
            for reference in self._offsets_by_context.get(context_id, ()):
                if reference.view not in allowed_views:
                    continue
                fallback_references[context_id].setdefault(reference.view, reference)
            references_by_chunk.update(
                {
                    reference.chunk_id: reference
                    for reference in fallback_references[context_id].values()
                }
            )
        references = tuple(
            sorted(references_by_chunk.values(), key=lambda item: item.corpus_order)
        )
        chunks_by_id = {
            chunk.chunk_id: chunk for chunk in self._read_references(references)
        }
        text_by_chunk = {
            chunk_id: chunks_by_id[chunk_id]
            for chunk_id in keep_chunks
            if chunk_id in chunks_by_id
        }
        fallback_by_context: dict[str, dict[str, CanonicalChunk]] = defaultdict(dict)
        for context_id, by_view in fallback_references.items():
            for view, reference in by_view.items():
                fallback_by_context[context_id][view] = chunks_by_id[reference.chunk_id]
        return _StoreMaterial(
            text_by_chunk=text_by_chunk,
            fallback_by_context=fallback_by_context,
            missing_requested_chunk_ids=tuple(
                sorted(requested_chunk_ids - set(text_by_chunk))
            ),
        )

def _question_term_counts(question: str) -> Counter:
    return Counter(
        token
        for token in expand_legal_abbreviations(word_tokens(question))
        if token not in VIETNAMESE_QUERY_STOPWORDS
    )


def _reorder_requested_by_question(
    requested_pair: Sequence[tuple[str, tuple[EvidenceAttribution, ...]]],
    question: str,
    text_by_chunk: Mapping[str, Any],
    max_chunks: int,
    min_gain: int,
) -> Sequence[tuple[str, tuple[EvidenceAttribution, ...]]]:
    if min_gain <= 0 or len(requested_pair) <= max_chunks:
        return requested_pair
    counts = _question_term_counts(question)
    if not counts:
        return requested_pair

    def score(chunk_id: str) -> int:
        item = text_by_chunk.get(chunk_id)
        if item is None:
            return -1
        return sum(
            counts[token]
            for token in expand_legal_abbreviations(word_tokens(item.text))
            if token in counts
        )

    order = [chunk_id for chunk_id, _ in requested_pair]
    kept: list[str] = []
    seen_text: set[str] = set()
    for chunk_id in order:
        item = text_by_chunk.get(chunk_id)
        if item is None or item.text in seen_text:
            continue
        kept.append(chunk_id)
        seen_text.add(item.text)
        if len(kept) >= max_chunks:
            break
    dropped = [chunk_id for chunk_id in order if chunk_id not in kept]
    if not kept or not dropped:
        return requested_pair
    scores = {chunk_id: score(chunk_id) for chunk_id in order}
    if max(scores[c] for c in dropped) - max(scores[c] for c in kept) < min_gain:
        return requested_pair
    position = {chunk_id: index for index, chunk_id in enumerate(order)}
    return sorted(
        requested_pair,
        key=lambda row: (-scores[row[0]], position[row[0]]),
    )


class EvidenceResolver:

    @staticmethod
    def resolve(
        universe: CandidateUniverse,
        fused_hits: Mapping[PairKey, Mapping[str, Any]],
        branch_hits: Sequence[Mapping[PairKey, Mapping[str, Any]]],
        store: CanonicalChunkStore,
        policy: EvidencePolicy,
        availability: EvidenceAvailability | None = None,
        questions: Mapping[str, str] | None = None,
    ) -> EvidenceBatch:
        _trace(
            "evidence.start",
            mode=policy.mode,
            max_chunks=policy.max_chunks,
            question_order_min_gain=policy.question_order_min_gain,
            candidates=len(universe.candidates),
        )
        availability = availability or EvidenceResolver.collect_availability(
            universe, fused_hits, branch_hits, policy
        )
        requested = EvidenceSelector.select(universe, availability, policy)
        material = store.materialize(
            requested_chunk_ids={
                chunk_id for values in requested.values() for chunk_id, _ in values
            },
            candidate_context_ids={row.context_id for row in universe.candidates},
            policy=policy,
        )
        reorder_by_question = policy.question_order_min_gain > 0 and questions is not None
        records: list[EvidenceRecord] = []
        for candidate in universe.candidates:
            pair = (candidate.query_id, candidate.context_id)
            selected: list[CanonicalChunk] = []
            selected_attributes: list[tuple[EvidenceAttribution, ...]] = []
            selected_reasons: list[str] = []
            seen_text: set[str] = set()
            pair_requests = requested[pair]
            if reorder_by_question:
                pair_requests = _reorder_requested_by_question(
                    pair_requests,
                    questions.get(candidate.query_id, ""),
                    material.text_by_chunk,
                    policy.max_chunks,
                    policy.question_order_min_gain,
                )
            for chunk_id, attributions in pair_requests:
                item = material.text_by_chunk.get(chunk_id)
                if item is None or item.text in seen_text:
                    continue
                selected.append(
                    EvidenceResolver._extended(item, material.text_by_chunk, policy)
                )
                selected_attributes.append(attributions)
                selected_reasons.append(
                    "champion_v1_saved_evidence"
                    if policy.mode == "champion_v1"
                    else "canonical_retriever_evidence"
                )
                seen_text.add(item.text)
                if len(selected) >= policy.max_chunks:
                    break
            fallback_order = EvidenceResolver._fallback_order(policy)
            for view in fallback_order:
                if len(selected) >= policy.max_chunks:
                    break
                item = material.fallback_by_context.get(candidate.context_id, {}).get(view)
                if item is None or item.text in seen_text:
                    continue
                selected.append(
                    EvidenceResolver._extended(item, material.text_by_chunk, policy)
                )
                selected_attributes.append((EvidenceAttribution(None, None, None),))
                selected_reasons.append("canonical_corpus_fallback")
                seen_text.add(item.text)
            if not selected:
                raise ValueError(
                    "No corpus text for query/context pair "
                    f"query_id={candidate.query_id} context_id={candidate.context_id}"
                )
            for item, attributions, reason in zip(
                selected, selected_attributes, selected_reasons
            ):
                records.append(
                    EvidenceRecord(
                        query_id=candidate.query_id,
                        context_id=candidate.context_id,
                        chunk_id=item.chunk_id,
                        view=item.view,
                        text=item.text,
                        text_sha256=hashlib.sha256(item.text.encode("utf-8")).hexdigest(),
                        attributions=attributions,
                        resolution_reason=reason,
                        corpus_sha256=store.corpus_sha256,
                    )
                )
        rows = tuple(records)
        by_view: dict[str, int] = defaultdict(int)
        for record in rows:
            by_view[record.view] += 1
        batch = EvidenceBatch(records=rows, content_hash=semantic_sha256(rows))
        _trace(
            "evidence.complete",
            records=len(rows),
            views=dict(sorted(by_view.items())),
            evidence_hash=batch.content_hash,
        )
        return batch

    @staticmethod
    def collect_availability(
        universe: CandidateUniverse,
        fused_hits: Mapping[PairKey, Mapping[str, Any]],
        branch_hits: Sequence[Mapping[PairKey, Mapping[str, Any]]],
        policy: EvidencePolicy,
    ) -> EvidenceAvailability:
        if policy.mode == "champion_v1":
            requested = EvidenceResolver._champion_requests(
                universe, fused_hits, branch_hits
            )
        else:
            requested = EvidenceResolver._canonical_requests(
                universe, fused_hits, branch_hits
            )
        records = tuple(
            AvailableEvidence(
                query_id=query_id,
                context_id=context_id,
                chunk_id=chunk_id,
                attributions=attributions,
                discovery_order=discovery_order,
            )
            for candidate in universe.candidates
            for query_id, context_id in ((candidate.query_id, candidate.context_id),)
            for discovery_order, (chunk_id, attributions) in enumerate(
                requested[(query_id, context_id)]
            )
        )
        return EvidenceAvailability(records=records, content_hash=semantic_sha256(records))

    @staticmethod
    def _champion_requests(
        universe: CandidateUniverse,
        fused_hits: Mapping[PairKey, Mapping[str, Any]],
        branch_hits: Sequence[Mapping[PairKey, Mapping[str, Any]]],
    ) -> dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]]:
        requested: dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]] = {}
        for candidate in universe.candidates:
            pair = (candidate.query_id, candidate.context_id)
            scored: dict[str, float] = {}
            for source_hits in (fused_hits, *branch_hits):
                hit = source_hits.get(pair)
                if not hit:
                    continue
                for key, value in hit.items():
                    if not (key == "chunk_id" or key.endswith("_chunk_id")):
                        continue
                    if not isinstance(value, str) or not value or value in scored:
                        continue
                    raw_score = float(hit.get("raw_score", 0.0) or 0.0)
                    scored[value] = raw_score
            requested[pair] = [
                (chunk_id, (EvidenceAttribution(None, None, raw_score),))
                for chunk_id, raw_score in sorted(
                    scored.items(), key=lambda item: -item[1]
                )
            ]
        return requested

    @staticmethod
    def _canonical_requests(
        universe: CandidateUniverse,
        fused_hits: Mapping[PairKey, Mapping[str, Any]],
        branch_hits: Sequence[Mapping[PairKey, Mapping[str, Any]]],
    ) -> dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]]:
        requested: dict[PairKey, list[tuple[str, tuple[EvidenceAttribution, ...]]]] = {}
        for candidate in universe.candidates:
            pair = (candidate.query_id, candidate.context_id)
            sources_by_chunk: dict[str, list[EvidenceAttribution]] = {}


            for source_hits in (fused_hits, *branch_hits):
                hit = source_hits.get(pair)
                if not hit:
                    continue
                for key, value in hit.items():
                    if key == "chunk_id" or not key.endswith("_chunk_id"):
                        continue
                    if not isinstance(value, str) or not value:
                        continue
                    source = key[: -len("_chunk_id")]
                    attribution = EvidenceAttribution(
                        source=source,
                        source_rank=_optional_int(hit.get(f"{source}_rank")),
                        source_raw_score=_optional_float(
                            hit.get(f"{source}_raw_score")
                        ),
                    )
                    values = sources_by_chunk.setdefault(value, [])
                    if attribution not in values:
                        values.append(attribution)
            requested[pair] = [
                (chunk_id, tuple(attributions))
                for chunk_id, attributions in sources_by_chunk.items()
            ]
        return requested

    @staticmethod
    def _fallback_order(policy: EvidencePolicy) -> tuple[str, ...]:
        prefix = ("article",) if policy.use_article_view else ()
        metadata = ("metadata",) if policy.include_metadata else ()
        return (*prefix, *metadata, "packed", "recovery", "short")

    @staticmethod
    def _extended(
        item: CanonicalChunk,
        text_by_chunk: Mapping[str, CanonicalChunk],
        policy: EvidencePolicy,
    ) -> CanonicalChunk:
        if policy.neighbor_context == 0:
            return item
        previous, following = _chunk_neighbors(item.chunk_id, policy.neighbor_context)
        parts: list[str] = []
        for chunk_id in (*previous, item.chunk_id, *following):
            neighbour = text_by_chunk.get(chunk_id)
            text = neighbour.text if neighbour is not None else ""
            if chunk_id == item.chunk_id and not text:
                text = item.text
            if text and text not in parts:
                parts.append(text)
        return CanonicalChunk(
            context_id=item.context_id,
            chunk_id=item.chunk_id,
            view=item.view,
            text="\n".join(parts),
        )

class UnsupportedSignalError(ValueError):
    pass

class InvalidStageSignalError(UnsupportedSignalError):
    pass

class InvalidShortlistError(ValueError):
    pass

def _signal_id(signal: RankingSignal) -> str:
    return semantic_sha256({"kind": type(signal).__name__, "value": signal})

def _numeric_context_key(context_id: str) -> tuple[int, int | str]:
    return (0, int(context_id)) if context_id.isdigit() else (1, context_id)

@dataclass(frozen=True)
class ShortlistView:

    view_id: str
    parent_view_id: str
    selected_by_query: tuple[tuple[str, tuple[str, ...]], ...]
    selection_signal_ids: tuple[str, ...]

    def context_ids(self, query_id: str) -> tuple[str, ...]:
        for value_query_id, context_ids in self.selected_by_query:
            if value_query_id == query_id:
                return context_ids
        return ()

@dataclass(frozen=True)
class PointwiseProjection:

    raw_chunk_channel: str
    context_score_channel: str
    rank_channel: str
    rank_tie_break_channel: str
    score_scale: str
    reducer: str = "max"

    def __post_init__(self) -> None:
        if self.reducer != "max":
            raise ValueError("V1 pointwise projection only supports reducer=max")
        names = (
            self.raw_chunk_channel,
            self.context_score_channel,
            self.rank_channel,
        )
        if any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError("Pointwise projection channels must be distinct and non-empty")
        if not self.rank_tie_break_channel or not self.score_scale:
            raise ValueError("Pointwise projection requires tie-break channel and scale")

    @property
    def normalized_context_score_channel(self) -> str:
        return f"{self.context_score_channel}_zscore"

    @property
    def output_channels(self) -> tuple[str, str, str, str]:
        return (
            self.raw_chunk_channel,
            self.context_score_channel,
            self.normalized_context_score_channel,
            self.rank_channel,
        )

@dataclass(frozen=True)
class Judge:
    stage_name: str
    adapter_id: str
    input_view: str
    output_channels: tuple[str, ...]
    projection: PointwiseProjection | None = None
    strict_rank_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.stage_name or not self.adapter_id or not self.input_view:
            raise ValueError("Judge requires stage, adapter and input view")
        if not self.output_channels or any(not channel for channel in self.output_channels):
            raise ValueError("Judge requires non-empty output channels")
        if len(set(self.output_channels)) != len(self.output_channels):
            raise ValueError("Judge output channels must be unique")
        if not set(self.strict_rank_channels).issubset(self.output_channels):
            raise ValueError("strict rank channels must be Judge output channels")

@dataclass(frozen=True)
class Parallel:
    branches: tuple[Any, ...]

@dataclass(frozen=True)
class Select:
    stage_name: str
    input_view: str
    rank_channel: str
    limit: int

    def __post_init__(self) -> None:
        if not self.stage_name or not self.rank_channel:
            raise InvalidShortlistError("Select requires stage_name and rank_channel")
        if self.limit < 1:
            raise InvalidShortlistError("Select limit must be positive")

@dataclass(frozen=True)
class Cascade:
    steps: tuple[Any, ...]

RecipeStep = Judge | Parallel | Select | Cascade

@dataclass(frozen=True)
class CompiledRecipe:
    recipe: RecipeStep
    recipe_hash: str
    universe: CandidateUniverse
    universe_hash: str
    finalizer_requirements: tuple[str, ...]

def _recipe_selects(step: RecipeStep) -> tuple[Select, ...]:
    if isinstance(step, Select):
        return (step,)
    if isinstance(step, Judge):
        return ()
    if isinstance(step, Parallel):
        return tuple(select for branch in step.branches for select in _recipe_selects(branch))
    if isinstance(step, Cascade):
        return tuple(select for child in step.steps for select in _recipe_selects(child))
    raise TypeError(f"Unsupported RecipeStep: {type(step).__name__}")

def _recipe_judges(step: RecipeStep) -> tuple[Judge, ...]:
    if isinstance(step, Judge):
        return (step,)
    if isinstance(step, Select):
        return ()
    if isinstance(step, Parallel):
        return tuple(judge for branch in step.branches for judge in _recipe_judges(branch))
    if isinstance(step, Cascade):
        return tuple(judge for child in step.steps for judge in _recipe_judges(child))
    raise TypeError(f"Unsupported RecipeStep: {type(step).__name__}")

def _rank_rows(
    ledger: SignalLedgerSnapshot,
    channel: str,
    query_id: str,
    allowed_context_ids: Sequence[str],
) -> list[tuple[str, int, str]]:
    allowed = set(allowed_context_ids)
    rows: dict[str, tuple[int, str]] = {}
    for signal in ledger.signals:
        if not isinstance(signal, RankSignal):
            continue
        if signal.channel != channel or signal.query_id != query_id:
            continue
        if signal.context_id not in allowed:
            continue
        if signal.context_id in rows:
            raise UnsupportedSignalError(
                f"Ambiguous rank signals for channel={channel} "
                f"query_id={query_id} context_id={signal.context_id}"
            )
        rows[signal.context_id] = (signal.rank, _signal_id(signal))
    missing = [context_id for context_id in allowed_context_ids if context_id not in rows]
    if missing:
        raise UnsupportedSignalError(
            f"Missing rank channel={channel} for query_id={query_id}: {missing[:5]}"
        )
    return sorted(
        (
            (context_id, rank, signal_id)
            for context_id, (rank, signal_id) in rows.items()
        ),
        key=lambda row: (row[1], _numeric_context_key(row[0])),
    )

class RecipeCompiler:

    @staticmethod
    def compile(
        recipe: RecipeStep,
        universe: CandidateUniverse,
        ledger: SignalLedgerSnapshot,
        adapter_descriptors: Mapping[str, Any],
        finalizer_requirements: Sequence[str],
    ) -> CompiledRecipe:
        unique_models = {
            descriptor.checkpoint_sha256: descriptor
            for judge in _recipe_judges(recipe)
            if (descriptor := adapter_descriptors.get(judge.adapter_id)) is not None
        }
        total_parameters = sum(
            int(descriptor.parameter_count) for descriptor in unique_models.values()
        )
        if total_parameters > 4_000_000_000:
            raise ValueError(
                "Recipe exceeds the rerank-stage 4B unique checkpoint parameter "
                "budget (rerank Judges only, not the whole-system competition "
                f"budget): {total_parameters}"
            )
        stage_names: set[str] = set()
        initial_channels = {
            signal.channel for signal in ledger.signals if hasattr(signal, "channel")
        }
        declared_channels = set(initial_channels)

        def compile_step(
            step: RecipeStep,
            available_views: set[str],
            available_channels: set[str],
        ) -> tuple[set[str], set[str]]:
            if isinstance(step, Judge):
                if step.stage_name in stage_names:
                    raise ValueError(f"duplicate recipe stage_name={step.stage_name}")
                stage_names.add(step.stage_name)
                if step.input_view not in available_views:
                    raise InvalidShortlistError(
                        f"Judge {step.stage_name} references unknown view {step.input_view}"
                    )
                descriptor = adapter_descriptors.get(step.adapter_id)
                if descriptor is None:
                    raise ValueError(f"Unknown adapter_id={step.adapter_id}")
                if getattr(descriptor, "eligibility", None) == "teacher_only":
                    raise ValueError(
                        f"teacher_only adapter not deployable: {step.adapter_id}"
                    )
                output_kinds = tuple(getattr(descriptor, "output_kinds", ()))
                supported_kinds = {
                    "score", "rank", "preference", "judgement", "feature"
                }
                if not output_kinds or not set(output_kinds).issubset(supported_kinds):
                    raise UnsupportedSignalError(
                        f"Adapter {step.adapter_id} declares unsupported output kinds"
                    )
                if step.projection is not None:
                    if tuple(step.output_channels) != step.projection.output_channels:
                        raise ValueError(
                            f"Judge {step.stage_name} output_channels must match projection"
                        )
                    if "score" not in output_kinds:
                        raise UnsupportedSignalError(
                            f"Adapter {step.adapter_id} cannot produce pointwise scores"
                        )
                    if step.strict_rank_channels:
                        raise ValueError(
                            "Pointwise projection owns its deterministic rank channel"
                        )
                elif step.strict_rank_channels and "rank" not in output_kinds:
                    raise UnsupportedSignalError(
                        f"Adapter {step.adapter_id} cannot produce strict ranks"
                    )
                for channel in step.output_channels:
                    if channel in declared_channels:
                        raise ValueError(f"duplicate output channel={channel}")
                    declared_channels.add(channel)
                    available_channels.add(channel)
                return available_views, available_channels
            if isinstance(step, Select):
                if step.stage_name in stage_names:
                    raise ValueError(f"duplicate recipe stage_name={step.stage_name}")
                stage_names.add(step.stage_name)
                if step.input_view not in available_views:
                    raise InvalidShortlistError(
                        f"Select {step.stage_name} references unknown view "
                        f"{step.input_view}"
                    )
                if step.rank_channel not in available_channels:
                    raise UnsupportedSignalError(
                        f"Select {step.stage_name} lacks rank channel={step.rank_channel}"
                    )
                return available_views | {step.stage_name}, available_channels
            if isinstance(step, Cascade):
                current_views, current_channels = available_views, available_channels
                for child in step.steps:
                    current_views, current_channels = compile_step(
                        child, current_views, current_channels
                    )
                return current_views, current_channels
            if isinstance(step, Parallel):
                merged_views, merged_channels = set(available_views), set(available_channels)
                for branch in step.branches:
                    branch_views, branch_channels = compile_step(
                        branch, set(available_views), set(available_channels)
                    )
                    merged_views.update(branch_views)
                    merged_channels.update(branch_channels)
                return merged_views, merged_channels
            raise TypeError(f"Unsupported RecipeStep: {type(step).__name__}")

        _, output_channels = compile_step(recipe, {"universe"}, set(initial_channels))
        missing = sorted(set(finalizer_requirements) - output_channels)
        if missing:
            raise UnsupportedSignalError(
                f"Finalizer missing required channels: {missing}"
            )
        return CompiledRecipe(
            recipe=recipe,
            recipe_hash=semantic_sha256(recipe),
            universe=universe,
            universe_hash=universe.content_hash,
            finalizer_requirements=tuple(finalizer_requirements),
        )

@dataclass(frozen=True)
class RankChannel:
    channel: str
    weight: float

    def __post_init__(self) -> None:
        if not self.channel or not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("RankChannel requires non-negative finite weight")

@dataclass(frozen=True)
class FinalRankingRow:
    query_id: str
    context_id: str
    final_score: float
    final_rank: int

@dataclass(frozen=True)
class FinalRanking:
    rows: tuple[FinalRankingRow, ...]

    def context_ids(self, query_id: str) -> tuple[str, ...]:
        return tuple(row.context_id for row in self.rows if row.query_id == query_id)

def _ranking_from_scores(
    universe: CandidateUniverse,
    scores: Mapping[PairKey, float],
    tie_break_ranks: Mapping[tuple[str, str], int] | None = None,
) -> FinalRanking:
    rows: list[FinalRankingRow] = []
    for query_id in universe.query_order:
        query_rows: list[tuple[str, float]] = []
        for context_id in universe.context_ids(query_id):
            value = scores.get((query_id, context_id))
            if value is None or not math.isfinite(value):
                raise UnsupportedSignalError(
                    f"Finalizer has no finite score for query_id={query_id} "
                    f"context_id={context_id}"
                )
            query_rows.append((context_id, value))
        query_rows.sort(
            key=lambda row: (
                -row[1],
                (tie_break_ranks or {}).get((query_id, row[0]), math.inf),
                _numeric_context_key(row[0]),
            )
        )
        rows.extend(
            FinalRankingRow(query_id, context_id, score, rank)
            for rank, (context_id, score) in enumerate(query_rows, 1)
        )
    return FinalRanking(rows=tuple(rows))

@dataclass(frozen=True)
class RrfFinalizer:
    channels: tuple[RankChannel, ...]
    rrf_k: int
    tie_break_channels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.channels or self.rrf_k < 0:
            raise ValueError("RrfFinalizer requires channels and non-negative rrf_k")
        if len({item.channel for item in self.channels}) != len(self.channels):
            raise ValueError("RrfFinalizer channels must be unique")

    def finalize(
        self, universe: CandidateUniverse, ledger: SignalLedgerSnapshot
    ) -> FinalRanking:
        ranks_by_channel: dict[str, dict[PairKey, int]] = {}
        for item in self.channels:
            values: dict[PairKey, int] = {}
            for query_id in universe.query_order:
                for context_id, rank, _ in _rank_rows(
                    ledger, item.channel, query_id, universe.context_ids(query_id)
                ):
                    values[(query_id, context_id)] = rank
            ranks_by_channel[item.channel] = values
        scores = {
            (candidate.query_id, candidate.context_id): sum(
                channel.weight
                / (self.rrf_k + ranks_by_channel[channel.channel][
                    (candidate.query_id, candidate.context_id)
                ])
                for channel in self.channels
            )
            for candidate in universe.candidates
        }
        tie_break_ranks: dict[PairKey, int] = {}
        for channel in self.tie_break_channels:
            if channel not in ranks_by_channel:
                raise UnsupportedSignalError(
                    f"RRF tie-break channel must be consumed: {channel}"
                )
            for pair, rank in ranks_by_channel[channel].items():
                tie_break_ranks.setdefault(pair, rank)
        return _ranking_from_scores(universe, scores, tie_break_ranks)

@dataclass(frozen=True)
class ScoreFusionFinalizer:
    channels: tuple[str, ...]

    def finalize(
        self, universe: CandidateUniverse, ledger: SignalLedgerSnapshot
    ) -> FinalRanking:
        if not self.channels:
            raise UnsupportedSignalError("ScoreFusionFinalizer requires channels")
        values: dict[str, dict[PairKey, ScoreSignal]] = {}
        scales: set[str] = set()
        for channel in self.channels:
            rows = {
                (signal.query_id, signal.context_id): signal
                for signal in ledger.signals
                if isinstance(signal, ScoreSignal)
                and signal.channel == channel
                and signal.entity_scope == "context"
            }
            expected = {
                (candidate.query_id, candidate.context_id) for candidate in universe.candidates
            }
            if set(rows) != expected:
                raise UnsupportedSignalError(
                    f"Score channel={channel} lacks complete context coverage"
                )
            channel_scales = {signal.scale for signal in rows.values()}
            if len(channel_scales) != 1:
                raise UnsupportedSignalError(
                    f"Score channel={channel} has inconsistent scale"
                )
            scales.update(channel_scales)
            values[channel] = rows
        if len(scales) != 1:
            raise UnsupportedSignalError(
                "Score fusion requires an explicit calibrator for mixed scales"
            )
        scores = {
            (candidate.query_id, candidate.context_id): sum(
                (
                    values[channel][(candidate.query_id, candidate.context_id)].value
                    if values[channel][(candidate.query_id, candidate.context_id)].orientation
                    == "higher_is_better"
                    else -values[channel][(candidate.query_id, candidate.context_id)].value
                )
                for channel in self.channels
            )
            for candidate in universe.candidates
        }
        return _ranking_from_scores(universe, scores)

def normalize_context_score_channel(
    values: Sequence[float], orientations: Sequence[str] | None = None
) -> tuple[tuple[float, ...], float, float, tuple[float, ...]]:
    if not values:
        raise ValueError("score channel cannot be empty")
    if orientations is None:
        orientations = tuple("higher_is_better" for _ in values)
    if len(values) != len(orientations):
        raise ValueError("values and orientations must have equal length")
    if any(item not in {"higher_is_better", "lower_is_better"} for item in orientations):
        raise ValueError("unsupported score orientation")
    oriented = tuple(
        value if orientation == "higher_is_better" else -value
        for value, orientation in zip(values, orientations, strict=True)
    )
    mean = sum(oriented) / len(oriented)
    variance = sum((value - mean) ** 2 for value in oriented) / len(oriented)
    standard_deviation = math.sqrt(variance)
    normalized = tuple(
        0.0 if standard_deviation == 0.0 else (value - mean) / standard_deviation
        for value in oriented
    )
    return oriented, mean, standard_deviation, normalized

def _complete_context_scores(
    universe: CandidateUniverse,
    ledger: SignalLedgerSnapshot,
    channel: str,
) -> dict[PairKey, ScoreSignal]:
    values: dict[PairKey, ScoreSignal] = {}
    for signal in ledger.signals:
        if not isinstance(signal, ScoreSignal):
            continue
        if signal.channel != channel or signal.entity_scope != "context":
            continue
        pair = (signal.query_id, signal.context_id)
        if pair in values:
            raise UnsupportedSignalError(
                f"Ambiguous context score channel={channel} pair={pair}"
            )
        values[pair] = signal
    expected = {(row.query_id, row.context_id) for row in universe.candidates}
    if set(values) != expected:
        raise UnsupportedSignalError(
            f"Score channel={channel} lacks complete context coverage"
        )
    return values

@dataclass(frozen=True)
class ConfidenceGateFinalizer:

    confidence_channel: str
    threshold: float
    high_confidence_finalizer: Any
    low_confidence_finalizer: Any

    def __post_init__(self) -> None:
        if not self.confidence_channel or not math.isfinite(self.threshold):
            raise ValueError("Confidence gate requires channel and finite threshold")
        if type(self.high_confidence_finalizer) is not type(self.low_confidence_finalizer):
            raise UnsupportedSignalError(
                "Confidence gate formulas must have the same finalizer type"
            )

    def finalize(
        self, universe: CandidateUniverse, ledger: SignalLedgerSnapshot
    ) -> FinalRanking:
        confidence = _complete_context_scores(universe, ledger, self.confidence_channel)
        if any(signal.orientation != "higher_is_better" for signal in confidence.values()):
            raise UnsupportedSignalError(
                "Confidence gate requires higher_is_better confidence scores"
            )
        high = self.high_confidence_finalizer.finalize(universe, ledger)
        low = self.low_confidence_finalizer.finalize(universe, ledger)
        high_scores = {
            (row.query_id, row.context_id): row.final_score for row in high.rows
        }
        low_scores = {
            (row.query_id, row.context_id): row.final_score for row in low.rows
        }
        scores = {
            pair: (
                high_scores[pair]
                if confidence[pair].value >= self.threshold
                else low_scores[pair]
            )
            for pair in confidence
        }
        return _ranking_from_scores(universe, scores)

@dataclass(frozen=True)
class LinearLtrFinalizer:

    feature_weights: tuple[tuple[str, str, float], ...]
    intercept: float = 0.0
    missing_value_policy: str = "reject"

    def __post_init__(self) -> None:
        if self.missing_value_policy not in {"reject", "zero"}:
            raise ValueError("LTR missing_value_policy must be reject or zero")
        if not math.isfinite(self.intercept):
            raise ValueError("LTR intercept must be finite")
        schema = [(name, version) for name, version, _ in self.feature_weights]
        if len(set(schema)) != len(schema):
            raise ValueError("LTR feature schema has duplicate name/version")
        if any(not math.isfinite(weight) for _, _, weight in self.feature_weights):
            raise ValueError("LTR feature weights must be finite")

    def finalize(
        self, universe: CandidateUniverse, ledger: SignalLedgerSnapshot
    ) -> FinalRanking:
        values: dict[tuple[str, str, str, str], float | None] = {}
        for signal in ledger.signals:
            if not isinstance(signal, FeatureSignal):
                continue
            key = (
                signal.query_id,
                signal.context_id,
                signal.feature_name,
                signal.feature_version,
            )
            if key in values:
                raise UnsupportedSignalError(f"Ambiguous LTR feature={key}")
            values[key] = signal.value
        scores: dict[PairKey, float] = {}
        for candidate in universe.candidates:
            pair = (candidate.query_id, candidate.context_id)
            score = self.intercept
            for name, version, weight in self.feature_weights:
                value = values.get((pair[0], pair[1], name, version))
                if value is None:
                    if self.missing_value_policy == "reject":
                        raise UnsupportedSignalError(
                            f"Missing LTR feature name={name} version={version} pair={pair}"
                        )
                    value = 0.0
                score += weight * value
            scores[pair] = score
        return _ranking_from_scores(universe, scores)

class NonFiniteSignalError(ValueError):
    pass

@dataclass(frozen=True)
class RunIdentity:
    run_id: str
    pipeline_stage: str
    evaluation_split: str
    host_account: str

@dataclass(frozen=True)
class JudgeDescriptor:
    adapter_id: str
    adapter_version: str
    model_class: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    config_sha256: str
    output_kinds: tuple[str, ...]
    eligibility: str
    parameter_count: int

    def __post_init__(self) -> None:
        if not self.adapter_id or not self.adapter_version or not self.model_class:
            raise ValueError("JudgeDescriptor requires adapter identity and model class")
        if self.eligibility not in {"production", "teacher_only"}:
            raise ValueError("JudgeDescriptor eligibility is invalid")
        if self.parameter_count < 0:
            raise ValueError("JudgeDescriptor parameter_count must be non-negative")

@dataclass(frozen=True)
class JudgeRuntime:
    device: str
    batch_size: int
    max_length: int = 2304
    bf16: bool = False
    attn_implementation: str = "sdpa"

    def __post_init__(self) -> None:
        if not self.device or self.batch_size < 1 or self.max_length < 1:
            raise ValueError("JudgeRuntime requires device, positive batch_size and max_length")
        if self.attn_implementation not in {"sdpa", "eager"}:
            raise ValueError("Unsupported attention implementation")

class StageChunkAccess:

    def __init__(
        self,
        store: CanonicalChunkStore,
        allowed_pairs: Sequence[tuple[str, str]],
    ) -> None:
        self._store = store
        self._allowed_pairs = frozenset(
            (str(query_id), str(context_id))
            for query_id, context_id in allowed_pairs
        )
        self._allowed_context_ids = frozenset(
            context_id for _, context_id in self._allowed_pairs
        )
        self._consumed: list[str] = []
        self._consumed_set: set[str] = set()
        self._consumed_chunk_pairs: list[tuple[str, str]] = []
        self._consumed_chunk_pair_set: set[tuple[str, str]] = set()
        self._consumed_pairs: list[tuple[str, str]] = []
        self._consumed_pair_set: set[tuple[str, str]] = set()

    @property
    def corpus_sha256(self) -> str:
        return self._store.corpus_sha256

    @property
    def consumed_chunk_ids(self) -> tuple[str, ...]:
        return tuple(self._consumed)

    @property
    def consumed_chunk_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._consumed_chunk_pairs)

    @property
    def consumed_context_pairs(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._consumed_pairs)

    def _record(self, query_id: str, chunks: Sequence[CanonicalChunk]) -> None:
        query_id = str(query_id)
        for chunk in chunks:
            pair = (query_id, chunk.context_id)
            if pair not in self._allowed_pairs:
                raise InvalidStageSignalError(
                    "canonical chunk access is outside current shortlist: "
                    f"query_id={query_id} context_id={chunk.context_id}"
                )
            if pair not in self._consumed_pair_set:
                self._consumed_pair_set.add(pair)
                self._consumed_pairs.append(pair)
            if chunk.chunk_id not in self._consumed_set:
                self._consumed_set.add(chunk.chunk_id)
                self._consumed.append(chunk.chunk_id)
            chunk_pair = (query_id, chunk.chunk_id)
            if chunk_pair not in self._consumed_chunk_pair_set:
                self._consumed_chunk_pair_set.add(chunk_pair)
                self._consumed_chunk_pairs.append(chunk_pair)

    def get_chunk(self, query_id: str, chunk_id: str) -> CanonicalChunk:
        chunk = self._store.get_chunk(chunk_id)
        self._record(query_id, (chunk,))
        return chunk

    def get_context_chunks(
        self,
        query_id: str,
        context_id: str,
        views: Sequence[str] | None = None,
    ) -> tuple[CanonicalChunk, ...]:
        if (str(query_id), str(context_id)) not in self._allowed_pairs:
            raise InvalidStageSignalError(
                "canonical context access is outside the selected query pair: "
                f"query_id={query_id} context_id={context_id}"
            )
        chunks = self._store.get_context_chunks(context_id, views)
        self._record(query_id, chunks)
        return chunks

@dataclass(frozen=True)
class JudgeStageInput:
    stage: Judge
    questions: Mapping[str, str]
    evidence: EvidenceBatch
    evidence_availability: EvidenceAvailability
    chunk_store: StageChunkAccess
    selected_pairs: tuple[PairKey, ...]
    stage_input_hash: str
    provenance: SignalProvenance

@dataclass(frozen=True)
class JudgeStageOutput:
    signals: tuple[RankingSignal, ...]
    consumed_chunk_ids: tuple[str, ...] = ()
    consumed_chunk_pairs: tuple[tuple[str, str], ...] = ()

    @classmethod
    def chunk_score_values(
        cls,
        stage_input: JudgeStageInput,
        values: Sequence[float],
    ) -> "JudgeStageOutput":
        projection = stage_input.stage.projection
        if projection is None:
            raise ValueError("Pointwise Judge requires a projection")
        if len(values) != len(stage_input.evidence.records):
            raise ValueError("Pointwise output count differs from evidence count")
        signals: list[ScoreSignal] = []
        for evidence, raw_value in zip(stage_input.evidence.records, values):
            value = float(raw_value)
            if not math.isfinite(value):
                raise NonFiniteSignalError(
                    f"Non-finite pointwise output stage={stage_input.stage.stage_name} "
                    f"query_id={evidence.query_id} context_id={evidence.context_id} "
                    f"chunk_id={evidence.chunk_id}"
                )
            signals.append(
                ScoreSignal(
                    query_id=evidence.query_id,
                    context_id=evidence.context_id,
                    chunk_id=evidence.chunk_id,
                    value=value,
                    channel=projection.raw_chunk_channel,
                    view_id=stage_input.stage.input_view,
                    scale=projection.score_scale,
                    orientation="higher_is_better",
                    entity_scope="chunk",
                    provenance=stage_input.provenance,
                )
            )
        return cls(
            signals=tuple(signals),
            consumed_chunk_ids=tuple(
                evidence.chunk_id for evidence in stage_input.evidence.records
            ),
            consumed_chunk_pairs=tuple(
                (evidence.query_id, evidence.chunk_id)
                for evidence in stage_input.evidence.records
            ),
        )

@dataclass(frozen=True)
class StageExecutionRecord:
    stage_name: str
    input_view: str
    stage_input_hash: str
    corpus_sha256: str
    evidence_hash: str
    evidence_availability_hash: str
    adapter_config_hash: str
    raw_cache_key: str
    cache_hit: bool
    derived_signal_cache_key: str | None
    derived_signal_cache_hit: bool | None
    consumed_chunk_ids: tuple[str, ...]
    consumed_chunk_pairs: tuple[tuple[str, str], ...]
    output_signal_hash: str
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None
    cuda_telemetry_unavailable_reason: str | None

@dataclass(frozen=True)
class _CudaTelemetryProbe:

    torch_module: object | None
    device: object | None
    unavailable_reason: str | None = None

def _cuda_telemetry_device(torch_module: object, runtime: JudgeRuntime) -> object:
    try:
        device = torch_module.device(runtime.device)
    except (TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Invalid CUDA JudgeRuntime device {runtime.device!r} for telemetry"
        ) from exc
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError(
            f"CUDA JudgeRuntime requires a CUDA device, got {runtime.device!r}"
        )
    device_index = 0 if getattr(device, "index", None) is None else device.index
    visible_device_count = int(torch_module.cuda.device_count())
    if device_index < 0 or device_index >= visible_device_count:
        raise RuntimeError(
            "CUDA JudgeRuntime device is outside the visible device range: "
            f"device={runtime.device!r}, visible_device_count={visible_device_count}"
        )
    return device

def _reset_cuda_peak_memory(runtime: JudgeRuntime) -> _CudaTelemetryProbe | None:
    if not runtime.device.startswith("cuda"):
        return None
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA JudgeRuntime requested but CUDA is unavailable")
    device = _cuda_telemetry_device(torch, runtime)
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError as exc:
        return _CudaTelemetryProbe(
            torch_module=None,
            device=None,
            unavailable_reason=(
                "reset_peak_memory_stats failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
    return _CudaTelemetryProbe(torch_module=torch, device=device)

def _cuda_peak_memory_bytes(
    probe: _CudaTelemetryProbe | None,
) -> tuple[int | None, int | None]:
    if probe is None or probe.unavailable_reason is not None:
        return None, None
    assert probe.torch_module is not None and probe.device is not None
    probe.torch_module.cuda.synchronize(probe.device)
    return (
        int(probe.torch_module.cuda.max_memory_allocated(probe.device)),
        int(probe.torch_module.cuda.max_memory_reserved(probe.device)),
    )

class JudgeSession(Protocol):
    def produce(self, stage_input: JudgeStageInput) -> JudgeStageOutput: ...

    def close(self) -> None: ...

class JudgeAdapter(Protocol):
    def describe(self) -> JudgeDescriptor: ...

    def open(self, runtime: JudgeRuntime) -> JudgeSession: ...

@dataclass(frozen=True)
class RerankRequest:

    run_identity: RunIdentity
    questions: Mapping[str, str]
    candidate_universe: CandidateUniverse
    initial_signals: tuple[RankingSignal, ...]
    chunk_store: CanonicalChunkStore
    fused_hits: Mapping[PairKey, Mapping[str, Any]]
    branch_hits: tuple[Mapping[PairKey, Mapping[str, Any]], ...]
    evidence_policy: EvidencePolicy
    recipe: RecipeStep
    finalizer: Any
    output_dir: Path
    runtime: JudgeRuntime
    cache: "ContentAddressedCache | None" = None
    input_provenance: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    producer_code_sha256: str | None = None
    runtime_closure: Mapping[str, Any] | None = None
    complete_system_budget: ProjectedSystemBudget | None = None

@dataclass(frozen=True)
class RerankRunResult:
    output_dir: Path
    final_ranking: FinalRanking
    evidence: EvidenceBatch
    ledger: SignalLedgerSnapshot
    stage_executions: tuple[StageExecutionRecord, ...]

class ArtifactWriter:

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        if self.output_dir.exists():
            raise FileExistsError(self.output_dir)
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        self._temporary = Path(
            tempfile.mkdtemp(
                prefix=f".{self.output_dir.name}.building.", dir=self.output_dir.parent
            )
        )
        self._closed = False
        self._promoted = False

    def write_json(self, relative_path: str, value: Any) -> None:
        destination = self._temporary / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(_semantic_value(value), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_jsonl(self, relative_path: str, values: Sequence[Any]) -> None:
        destination = self._temporary / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                handle.write(json.dumps(_semantic_value(value), ensure_ascii=False))
                handle.write("\n")

    def commit(self, complete_metadata: Mapping[str, Any]) -> Path:
        if self._closed:
            raise RuntimeError("ArtifactWriter is already closed")
        os.replace(self._temporary, self.output_dir)
        self._promoted = True
        complete_path = self.output_dir / "complete.json"
        temporary_complete = self.output_dir / ".complete.json.building"
        try:
            temporary_complete.write_text(
                json.dumps(_semantic_value(complete_metadata), ensure_ascii=False, indent=2)
                + "\n",
                encoding="utf-8",
            )
            os.replace(temporary_complete, complete_path)
        except Exception:
            temporary_complete.unlink(missing_ok=True)
            shutil.rmtree(self.output_dir)
            raise
        self._closed = True
        return self.output_dir

    def abort(self) -> None:
        if self._closed:
            return
        if self._promoted:
            if self.output_dir.is_dir() and not (self.output_dir / "complete.json").is_file():
                shutil.rmtree(self.output_dir)
            self._closed = True
            return
        if self._temporary.is_dir() and self._temporary.name.startswith(
            f".{self.output_dir.name}.building."
        ):
            shutil.rmtree(self._temporary)
        self._closed = True

class ContentAddressedCache:

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _entry(self, namespace: str, key: str) -> Path:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", namespace):
            raise ValueError(f"Invalid cache namespace={namespace}")
        if not re.fullmatch(r"[0-9a-f]{64}", key):
            raise ValueError("Cache key must be a SHA-256 hex digest")
        return self.root / namespace / key

    def load_json(self, namespace: str, key: str) -> Mapping[str, Any] | None:
        entry = self._entry(namespace, key)
        payload_path = entry / "payload.json"
        complete_path = entry / "complete.json"
        if not payload_path.is_file() or not complete_path.is_file():
            return None
        try:
            payload_bytes = payload_path.read_bytes()
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, Mapping) or not isinstance(complete, Mapping):
            return None
        if complete.get("key") != key:
            return None
        if complete.get("payload_sha256") != hashlib.sha256(payload_bytes).hexdigest():
            return None
        return payload

    def store_json(self, namespace: str, key: str, payload: Mapping[str, Any]) -> None:
        entry = self._entry(namespace, key)
        if self.load_json(namespace, key) is not None:
            return
        entry.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{key}.building.", dir=entry.parent))
        try:
            payload_bytes = (
                json.dumps(_semantic_value(payload), ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode("utf-8")
            (temporary / "payload.json").write_bytes(payload_bytes)
            (temporary / "complete.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "key": key,
                        "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                os.replace(temporary, entry)
            except FileExistsError:
                pass
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def evidence_key(
        universe: CandidateUniverse,
        store: CanonicalChunkStore,
        policy: EvidencePolicy,
        availability: EvidenceAvailability,
        *,
        producer_code_sha256: str,
    ) -> str:
        return semantic_sha256(
            {
                "schema": "evidence-v1",
                "universe": universe.content_hash,
                "corpus": store.corpus_sha256,
                "policy": policy,
                "evidence_availability": availability.content_hash,
                "producer_code_sha256": producer_code_sha256,
            }
        )

    @staticmethod
    def raw_inference_key(
        stage_input: JudgeStageInput,
        descriptor: JudgeDescriptor,
        runtime: JudgeRuntime,
        *,
        producer_code_sha256: str,
    ) -> str:
        return semantic_sha256(
            {
                "schema": "raw-inference-v3",
                "stage_input_hash": stage_input.stage_input_hash,
                "descriptor": descriptor,
                "runtime": runtime,
                "producer_code_sha256": producer_code_sha256,
            }
        )

    @staticmethod
    def derived_signal_key(
        raw_signal_hash: str,
        projection: PointwiseProjection,
        *,
        stage_input_hash: str,
        rank_tie_break_hash: str,
        producer_code_sha256: str,
    ) -> str:
        return semantic_sha256(
            {
                "schema": "derived-signal-v1",
                "raw_signal_hash": raw_signal_hash,
                "projection": projection,
                "stage_input_hash": stage_input_hash,
                "rank_tie_break_hash": rank_tie_break_hash,
                "producer_code_sha256": producer_code_sha256,
            }
        )

    @staticmethod
    def finalizer_key(
        universe: CandidateUniverse,
        ledger: SignalLedgerSnapshot,
        finalizer: Any,
        *,
        producer_code_sha256: str,
    ) -> str:
        return semantic_sha256(
            {
                "schema": "finalizer-v1",
                "universe": universe.content_hash,
                "ledger": ledger.content_hash,
                "finalizer": finalizer,
                "producer_code_sha256": producer_code_sha256,
            }
        )

def _finalizer_requirements(finalizer: Any) -> tuple[str, ...]:
    if isinstance(finalizer, RrfFinalizer):
        return tuple(channel.channel for channel in finalizer.channels)
    if isinstance(finalizer, ScoreFusionFinalizer):
        return finalizer.channels
    if isinstance(finalizer, ConfidenceGateFinalizer):
        return (
            finalizer.confidence_channel,
            *_finalizer_requirements(finalizer.high_confidence_finalizer),
            *_finalizer_requirements(finalizer.low_confidence_finalizer),
        )
    if isinstance(finalizer, LinearLtrFinalizer):
        return ()
    raise TypeError(f"Unsupported Finalizer: {type(finalizer).__name__}")

def _stage_provenance(stage: Judge, descriptor: JudgeDescriptor) -> SignalProvenance:
    return SignalProvenance(
        producer_stage=stage.stage_name,
        producer_type="judge_adapter",
        source=descriptor.adapter_id,
        config_hash=semantic_sha256(
            {"descriptor": descriptor, "stage": stage, "projection": stage.projection}
        ),
        adapter_id=descriptor.adapter_id,
        adapter_version=descriptor.adapter_version,
        model_manifest_sha256=descriptor.config_sha256,
        input_artifact_hashes=(("recipe_stage", semantic_sha256(stage)),),
        produced_at_utc=datetime.now(UTC).isoformat(),
    )

def _ranking_signals_from_cache(
    payload: Mapping[str, Any],
    *,
    schema: str | None = None,
) -> tuple[RankingSignal, ...]:
    supported_schemas = (
        {schema}
        if schema is not None
        else {"raw-ranking-signals-v2", "raw-ranking-signals-v3"}
    )
    if payload.get("schema") not in supported_schemas:
        raise ValueError(f"Unsupported ranking-signal cache schema={payload.get('schema')}")
    values = payload.get("signals")
    if not isinstance(values, list):
        raise ValueError("Raw inference cache lacks signals")
    signal_classes: Mapping[str, Any] = {
        "ScoreSignal": ScoreSignal,
        "RankSignal": RankSignal,
        "PreferenceSignal": PreferenceSignal,
        "JudgementSignal": JudgementSignal,
        "FeatureSignal": FeatureSignal,
        "SelectionSignal": SelectionSignal,
    }
    signals: list[RankingSignal] = []
    for row in values:
        if not isinstance(row, Mapping) or row.get("kind") not in signal_classes:
            raise ValueError("Raw inference cache has malformed signal")
        signal = dict(row)
        signal_class = signal_classes[str(signal.pop("kind"))]
        claimed_signal_id = signal.pop("signal_id", None)
        provenance = signal.pop("provenance", None)
        if not isinstance(provenance, Mapping):
            raise ValueError("Raw inference cache signal lacks provenance")
        parsed_signal = signal_class(
            **signal,
            provenance=SignalProvenance(
                producer_stage=str(provenance["producer_stage"]),
                producer_type=str(provenance["producer_type"]),
                source=str(provenance["source"]),
                config_hash=str(provenance["config_hash"]),
                adapter_id=str(provenance["adapter_id"]),
                adapter_version=str(provenance["adapter_version"]),
                model_manifest_sha256=str(provenance["model_manifest_sha256"]),
                input_artifact_hashes=tuple(
                    (str(name), str(value))
                    for name, value in provenance["input_artifact_hashes"]
                ),
                produced_at_utc=str(provenance["produced_at_utc"]),
            ),
        )
        if (
            claimed_signal_id is not None
            and claimed_signal_id != _signal_id(parsed_signal)
        ):
            raise ValueError(
                "Raw inference cache signal_id differs from reconstructed signal"
            )
        signals.append(parsed_signal)
    return tuple(signals)

def _evidence_cache_payload(
    evidence: EvidenceBatch,
    universe: CandidateUniverse,
    store: CanonicalChunkStore,
    policy: EvidencePolicy,
    availability: EvidenceAvailability,
    producer_code_sha256: str,
) -> Mapping[str, Any]:
    return {
        "schema": "evidence-batch-v1",
        "universe_hash": universe.content_hash,
        "corpus_sha256": store.corpus_sha256,
        "evidence_policy": _semantic_value(policy),
        "evidence_availability_hash": availability.content_hash,
        "producer_code_sha256": producer_code_sha256,
        "evidence_hash": evidence.content_hash,
        "records": [_semantic_value(record) for record in evidence.records],
    }

def _evidence_batch_from_cache(
    payload: Mapping[str, Any],
    universe: CandidateUniverse,
    store: CanonicalChunkStore,
    policy: EvidencePolicy,
    availability: EvidenceAvailability,
    producer_code_sha256: str,
) -> EvidenceBatch:
    if payload.get("schema") != "evidence-batch-v1":
        raise ValueError("Unsupported evidence cache schema")
    if (
        payload.get("universe_hash") != universe.content_hash
        or payload.get("corpus_sha256") != store.corpus_sha256
        or payload.get("evidence_availability_hash") != availability.content_hash
        or payload.get("producer_code_sha256") != producer_code_sha256
        or semantic_sha256(payload.get("evidence_policy")) != semantic_sha256(policy)
    ):
        raise ValueError("Evidence cache input identity mismatch")
    values = payload.get("records")
    if not isinstance(values, list):
        raise ValueError("Evidence cache lacks records")
    store.verify_current()
    expected_pairs = {
        (candidate.query_id, candidate.context_id) for candidate in universe.candidates
    }
    records: list[EvidenceRecord] = []
    count_by_pair: dict[PairKey, int] = defaultdict(int)
    for row in values:
        if not isinstance(row, Mapping):
            raise ValueError("Evidence cache has malformed record")
        try:
            attribution_rows = row["attributions"]
            if not isinstance(attribution_rows, list) or not attribution_rows:
                raise ValueError("Evidence cache record lacks attributions")
            attributions: list[EvidenceAttribution] = []
            for attribution in attribution_rows:
                if not isinstance(attribution, Mapping):
                    raise ValueError("Evidence cache has malformed attribution")
                source = attribution.get("source")
                source_rank = attribution.get("source_rank")
                source_raw_score = attribution.get("source_raw_score")
                if source is not None and (not isinstance(source, str) or not source):
                    raise ValueError("Evidence cache has invalid attribution source")
                if source_rank is not None and (
                    isinstance(source_rank, bool) or not isinstance(source_rank, int)
                ):
                    raise ValueError("Evidence cache has invalid attribution rank")
                if source_raw_score is not None and (
                    isinstance(source_raw_score, bool)
                    or not isinstance(source_raw_score, (int, float))
                    or not math.isfinite(float(source_raw_score))
                ):
                    raise ValueError("Evidence cache has invalid attribution raw score")
                if (
                    policy.mode == "canonical_evidence"
                    and source is not None
                    and source not in policy.source_priority
                ):
                    raise ValueError("Evidence cache source is absent from policy priority")
                attributions.append(
                    EvidenceAttribution(
                        source=source,
                        source_rank=source_rank,
                        source_raw_score=(
                            float(source_raw_score)
                            if source_raw_score is not None
                            else None
                        ),
                    )
                )
            record = EvidenceRecord(
                query_id=str(row["query_id"]),
                context_id=str(row["context_id"]),
                chunk_id=str(row["chunk_id"]),
                view=str(row["view"]),
                text=str(row["text"]),
                text_sha256=str(row["text_sha256"]),
                attributions=tuple(attributions),
                resolution_reason=str(row["resolution_reason"]),
                corpus_sha256=str(row["corpus_sha256"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Evidence cache has malformed record") from exc
        pair = (record.query_id, record.context_id)
        if pair not in expected_pairs:
            raise ValueError("Evidence cache record is outside Candidate Universe")
        if record.corpus_sha256 != store.corpus_sha256:
            raise ValueError("Evidence cache record has mismatched corpus hash")
        if not record.text or (
            hashlib.sha256(record.text.encode("utf-8")).hexdigest()
            != record.text_sha256
        ):
            raise ValueError("Evidence cache record text hash mismatch")
        if store.context_id_for_chunk(record.chunk_id) != record.context_id:
            raise ValueError("Evidence cache chunk does not belong to its context")
        count_by_pair[pair] += 1
        if count_by_pair[pair] > policy.max_chunks:
            raise ValueError("Evidence cache exceeds evidence policy chunk limit")
        records.append(record)
    if set(count_by_pair) != expected_pairs:
        raise ValueError("Evidence cache is incomplete for Candidate Universe")
    batch = EvidenceBatch(records=tuple(records), content_hash=semantic_sha256(tuple(records)))
    if payload.get("evidence_hash") != batch.content_hash:
        raise ValueError("Evidence cache content hash mismatch")
    return batch

def _final_ranking_from_cache(payload: Mapping[str, Any]) -> FinalRanking:
    if payload.get("schema") != "final-ranking-v1":
        raise ValueError("Unsupported finalizer cache schema")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Finalizer cache lacks rows")
    parsed: list[FinalRankingRow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Finalizer cache has malformed row")
        parsed.append(
            FinalRankingRow(
                query_id=str(row["query_id"]),
                context_id=str(row["context_id"]),
                final_score=float(row["final_score"]),
                final_rank=int(row["final_rank"]),
            )
        )
    return FinalRanking(rows=tuple(parsed))

def _filter_evidence(
    evidence: EvidenceBatch, selected_pairs: set[PairKey]
) -> EvidenceBatch:
    records = tuple(
        record
        for record in evidence.records
        if (record.query_id, record.context_id) in selected_pairs
    )
    if not records:
        raise ValueError("Judge view resolved no evidence")
    return EvidenceBatch(records=records, content_hash=semantic_sha256(records))

def _stage_input_identity(
    stage: Judge,
    selected_pairs: Sequence[PairKey],
    questions: Mapping[str, str],
    evidence: EvidenceBatch,
    availability: EvidenceAvailability,
    store: CanonicalChunkStore,
    evidence_policy: EvidencePolicy,
) -> str:
    return semantic_sha256(
        {
            "schema": "judge-stage-input-v1",
            "stage": stage,
            "selected_pairs": tuple(selected_pairs),
            "questions": tuple(
                (query_id, questions[query_id])
                for query_id in dict.fromkeys(query_id for query_id, _ in selected_pairs)
            ),
            "selected_evidence_hash": evidence.content_hash,
            "evidence_availability_hash": availability.content_hash,
            "corpus_sha256": store.corpus_sha256,
            "evidence_policy": evidence_policy,
        }
    )

def _validated_consumed_chunks(
    output: JudgeStageOutput,
    access: StageChunkAccess,
    store: CanonicalChunkStore,
    selected_pairs: set[PairKey],
) -> tuple[tuple[str, ...], tuple[PairKey, ...]]:
    store.verify_current()
    for query_id, context_id in access.consumed_context_pairs:
        if (query_id, context_id) not in selected_pairs:
            raise InvalidStageSignalError(
                "consumed chunk access is outside current shortlist: "
                f"query_id={query_id} context_id={context_id}"
            )
    chunk_pairs: list[PairKey] = [
        (str(query_id), str(chunk_id))
        for query_id, chunk_id in (
            *output.consumed_chunk_pairs,
            *access.consumed_chunk_pairs,
        )
    ]
    paired_chunk_ids = {chunk_id for _, chunk_id in chunk_pairs}
    signal_pairs_by_chunk: dict[str, set[PairKey]] = {}
    for signal in output.signals:
        chunk_id = getattr(signal, "chunk_id", None)
        if chunk_id is not None:
            signal_pairs_by_chunk.setdefault(str(chunk_id), set()).add(
                (str(signal.query_id), str(chunk_id))
            )
    for chunk_id in (*output.consumed_chunk_ids, *access.consumed_chunk_ids):
        chunk_id = str(chunk_id)
        if chunk_id in paired_chunk_ids:
            continue
        inferred_pairs = signal_pairs_by_chunk.get(chunk_id, set())
        if inferred_pairs:
            chunk_pairs.extend(sorted(inferred_pairs))
            paired_chunk_ids.add(chunk_id)
            continue
        context_id = store.context_id_for_chunk(chunk_id)
        matching_pairs = sorted(
            pair for pair in selected_pairs if pair[1] == context_id
        )
        if len(matching_pairs) != 1:
            raise InvalidStageSignalError(
                "consumed chunk provenance is ambiguous across queries: "
                f"chunk_id={chunk_id} context_id={context_id}"
            )
        chunk_pairs.append(matching_pairs[0])
        paired_chunk_ids.add(chunk_id)

    validated_pairs: list[PairKey] = []
    seen_pairs: set[PairKey] = set()
    consumed_ids: list[str] = []
    seen_ids: set[str] = set()
    for query_id, chunk_id in chunk_pairs:
        context_id = store.context_id_for_chunk(chunk_id)
        if (query_id, context_id) not in selected_pairs:
            raise InvalidStageSignalError(
                "consumed chunk is outside current shortlist: "
                f"query_id={query_id} chunk_id={chunk_id} context_id={context_id}"
            )
        pair = (query_id, chunk_id)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            validated_pairs.append(pair)
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            consumed_ids.append(chunk_id)
    return tuple(consumed_ids), tuple(validated_pairs)

def _validate_stage_signal_envelopes(
    stage: Judge,
    descriptor: JudgeDescriptor,
    stage_input: JudgeStageInput,
    output: JudgeStageOutput,
    selected_pairs: set[PairKey],
) -> tuple[RankingSignal, ...]:
    signals = tuple(output.signals)
    if not signals:
        raise InvalidStageSignalError(
            f"Judge {stage.stage_name} emitted no signals"
        )
    kind_by_class: Mapping[type[Any], str] = {
        ScoreSignal: "score",
        RankSignal: "rank",
        PreferenceSignal: "preference",
        JudgementSignal: "judgement",
        FeatureSignal: "feature",
    }
    for signal in signals:
        signal_kind = kind_by_class.get(type(signal))
        if signal_kind is None or signal_kind not in descriptor.output_kinds:
            raise InvalidStageSignalError(
                f"Judge {stage.stage_name} emitted undeclared signal kind"
            )
        if signal.channel not in stage.output_channels:
            raise InvalidStageSignalError(
                f"Judge {stage.stage_name} emitted unexpected channel={signal.channel}"
            )
        if signal.view_id != stage.input_view:
            raise InvalidStageSignalError(
                f"Judge {stage.stage_name} emitted signal for wrong view={signal.view_id}"
            )
        if signal.provenance != stage_input.provenance:
            raise InvalidStageSignalError(
                f"Judge {stage.stage_name} emitted mismatched provenance"
            )
        references = (
            (
                (signal.query_id, signal.left_context_id),
                (signal.query_id, signal.right_context_id),
            )
            if isinstance(signal, PreferenceSignal)
            else ((signal.query_id, signal.context_id),)
        )
        if any(reference not in selected_pairs for reference in references):
            raise InvalidStageSignalError(
                f"Judge {stage.stage_name} emitted signal outside current shortlist"
            )
    return signals

def _validate_direct_signals(
    stage: Judge,
    descriptor: JudgeDescriptor,
    stage_input: JudgeStageInput,
    output: JudgeStageOutput,
    selected_pairs: set[PairKey],
) -> tuple[RankingSignal, ...]:
    signals = _validate_stage_signal_envelopes(
        stage, descriptor, stage_input, output, selected_pairs
    )

    for channel in stage.strict_rank_channels:
        for query_id in sorted({query_id for query_id, _ in selected_pairs}):
            expected_contexts = {
                context_id
                for pair_query_id, context_id in selected_pairs
                if pair_query_id == query_id
            }
            rank_signals = tuple(
                signal
                for signal in signals
                if isinstance(signal, RankSignal)
                and signal.channel == channel
                and signal.query_id == query_id
            )
            actual_contexts = {signal.context_id for signal in rank_signals}
            ranks = sorted(signal.rank for signal in rank_signals)
            if (
                actual_contexts != expected_contexts
                or len(rank_signals) != len(expected_contexts)
                or ranks != list(range(1, len(expected_contexts) + 1))
            ):
                raise InvalidStageSignalError(
                    f"Judge {stage.stage_name} emitted invalid strict listwise "
                    f"ordering channel={channel} query_id={query_id}"
                )
    return signals

def _validated_pointwise_raw_signals(
    descriptor: JudgeDescriptor,
    stage_input: JudgeStageInput,
    output: JudgeStageOutput,
    selected_pairs: set[PairKey],
) -> tuple[tuple[ScoreSignal, ...], Mapping[PairKey, tuple[ScoreSignal, ...]]]:
    stage = stage_input.stage
    projection = stage.projection
    if projection is None:
        raise ValueError("Pointwise Judge requires a projection")
    stage_signals = _validate_stage_signal_envelopes(
        stage, descriptor, stage_input, output, selected_pairs
    )
    raw_by_pair: dict[PairKey, list[ScoreSignal]] = defaultdict(list)
    raw: list[ScoreSignal] = []
    for signal in stage_signals:
        if not isinstance(signal, ScoreSignal):
            raise UnsupportedSignalError(
                f"Judge {stage.stage_name} emitted non-score output for pointwise projection"
            )
        if signal.channel != projection.raw_chunk_channel:
            raise UnsupportedSignalError(
                f"Judge {stage.stage_name} emitted unexpected channel={signal.channel}"
            )
        if signal.entity_scope != "chunk" or not math.isfinite(signal.value):
            raise NonFiniteSignalError(
                f"Judge {stage.stage_name} emitted invalid raw chunk score"
            )
        pair = (signal.query_id, signal.context_id)
        if pair not in selected_pairs:
            raise UnsupportedSignalError(
                f"Judge {stage.stage_name} emitted a signal outside its input view"
            )
        raw_by_pair[pair].append(signal)
        raw.append(signal)
    missing = sorted(selected_pairs - set(raw_by_pair))
    if missing:
        raise UnsupportedSignalError(
            f"Judge {stage.stage_name} has no raw score for pairs={missing[:5]}"
        )
    return tuple(raw), {pair: tuple(values) for pair, values in raw_by_pair.items()}

def _pointwise_rank_tie_break_hash(
    universe: CandidateUniverse,
    existing: SignalLedgerSnapshot,
    projection: PointwiseProjection,
    selected_pairs: set[PairKey],
) -> str:
    rows_by_query: list[tuple[str, tuple[tuple[str, int, str], ...]]] = []
    for query_id in universe.query_order:
        view_contexts = tuple(
            context_id
            for context_id in universe.context_ids(query_id)
            if (query_id, context_id) in selected_pairs
        )
        if view_contexts:
            rows_by_query.append(
                (
                    query_id,
                    tuple(
                        _rank_rows(
                            existing,
                            projection.rank_tie_break_channel,
                            query_id,
                            view_contexts,
                        )
                    ),
                )
            )
    return semantic_sha256(tuple(rows_by_query))

def _derive_pointwise_signals(
    universe: CandidateUniverse,
    existing: SignalLedgerSnapshot,
    descriptor: JudgeDescriptor,
    stage_input: JudgeStageInput,
    output: JudgeStageOutput,
    selected_pairs: set[PairKey],
) -> tuple[RankingSignal, ...]:
    stage = stage_input.stage
    projection = stage.projection
    if projection is None:
        raise ValueError("Pointwise Judge requires a projection")
    raw, raw_by_pair = _validated_pointwise_raw_signals(
        descriptor, stage_input, output, selected_pairs
    )
    context_scores: dict[PairKey, ScoreSignal] = {}
    for pair, signals in raw_by_pair.items():
        best = signals[0]
        for candidate in signals[1:]:
            if candidate.value > best.value:
                best = candidate
        context_scores[pair] = ScoreSignal(
            query_id=best.query_id,
            context_id=best.context_id,
            chunk_id=best.chunk_id,
            value=best.value,
            channel=projection.context_score_channel,
            view_id=stage.input_view,
            scale=projection.score_scale,
            orientation="higher_is_better",
            entity_scope="context",
            provenance=best.provenance,
        )
    normalized_context_scores: list[ScoreSignal] = []
    for query_id in universe.query_order:
        rows = [
            (context_id, context_scores[(query_id, context_id)])
            for context_id in universe.context_ids(query_id)
            if (query_id, context_id) in context_scores
        ]
        if not rows:
            continue
        _, _, _, zscores = normalize_context_score_channel(
            tuple(score.value for _, score in rows),
            tuple(score.orientation for _, score in rows),
        )
        normalized_context_scores.extend(
            ScoreSignal(
                query_id=query_id,
                context_id=context_id,
                chunk_id=score.chunk_id,
                value=zscore,
                channel=projection.normalized_context_score_channel,
                view_id=stage.input_view,
                scale="zscore",
                orientation="higher_is_better",
                entity_scope="context",
                provenance=score.provenance,
            )
            for (context_id, score), zscore in zip(rows, zscores, strict=True)
        )
    tie_break: dict[PairKey, int] = {}
    for query_id in universe.query_order:
        view_contexts = tuple(
            context_id
            for context_id in universe.context_ids(query_id)
            if (query_id, context_id) in selected_pairs
        )
        if not view_contexts:
            continue
        for context_id, rank, _ in _rank_rows(
            existing, projection.rank_tie_break_channel, query_id, view_contexts
        ):
            tie_break[(query_id, context_id)] = rank
    ranks: list[RankSignal] = []
    for query_id in universe.query_order:
        rows = [
            (context_id, context_scores[(query_id, context_id)])
            for context_id in universe.context_ids(query_id)
            if (query_id, context_id) in context_scores
        ]
        rows.sort(
            key=lambda row: (
                -row[1].value,
                tie_break[(query_id, row[0])],
                _numeric_context_key(row[0]),
            )
        )
        ranks.extend(
            RankSignal(
                query_id=query_id,
                context_id=context_id,
                rank=rank,
                channel=projection.rank_channel,
                view_id=stage.input_view,
                provenance=context_score.provenance,
            )
            for rank, (context_id, context_score) in enumerate(rows, 1)
        )
    return (*raw, *context_scores.values(), *normalized_context_scores, *ranks)

def _select_view(
    universe: CandidateUniverse,
    ledger: TypedSignalLedger,
    select: Select,
    views: Mapping[str, ShortlistView],
) -> ShortlistView:
    parent = views.get(select.input_view)
    selected_by_query: list[tuple[str, tuple[str, ...]]] = []
    signal_ids: list[str] = []
    for query_id in universe.query_order:
        available = (
            parent.context_ids(query_id)
            if parent is not None
            else universe.context_ids(query_id)
        )
        rows = _rank_rows(ledger.snapshot(), select.rank_channel, query_id, available)
        selected = tuple(context_id for context_id, _, _ in rows[: select.limit])
        selected_by_query.append((query_id, selected))
        signal_ids.extend(signal_id for _, _, signal_id in rows[: select.limit])
    view = ShortlistView(
        view_id=select.stage_name,
        parent_view_id=select.input_view,
        selected_by_query=tuple(selected_by_query),
        selection_signal_ids=tuple(signal_ids),
    )
    provenance = SignalProvenance(
        producer_stage=select.stage_name,
        producer_type="select",
        source=select.rank_channel,
        config_hash=semantic_sha256(select),
        adapter_id="select",
        adapter_version="v1",
        model_manifest_sha256="not_applicable",
        input_artifact_hashes=(("selection_signals", semantic_sha256(signal_ids)),),
        produced_at_utc=datetime.now(UTC).isoformat(),
    )
    selected_pairs = {
        (query_id, context_id)
        for query_id, context_ids in view.selected_by_query
        for context_id in context_ids
    }
    ledger.append_batch(
        tuple(
            SelectionSignal(
                query_id=candidate.query_id,
                context_id=candidate.context_id,
                selected=(candidate.query_id, candidate.context_id) in selected_pairs,
                channel=f"{select.stage_name}_selection",
                view_id=select.stage_name,
                reason=f"top_{select.limit}_by_{select.rank_channel}",
                provenance=provenance,
            )
            for candidate in universe.candidates
        )
    )
    return view

def _execute_recipe_step(
    step: RecipeStep,
    universe: CandidateUniverse,
    ledger: TypedSignalLedger,
    views: dict[str, ShortlistView],
    questions: Mapping[str, str],
    evidence: EvidenceBatch,
    evidence_availability: EvidenceAvailability,
    chunk_store: CanonicalChunkStore,
    evidence_policy: EvidencePolicy,
    adapters: Mapping[str, JudgeAdapter],
    descriptors: Mapping[str, JudgeDescriptor],
    runtime: JudgeRuntime,
    cache: ContentAddressedCache | None,
    producer_code_sha256: str,
    execution_records: list[StageExecutionRecord],
) -> None:
    if isinstance(step, Judge):
        source_view = views.get(step.input_view)
        selected_pairs = {
            (query_id, context_id)
            for query_id in universe.query_order
            for context_id in (
                source_view.context_ids(query_id)
                if source_view is not None
                else universe.context_ids(query_id)
            )
        }
        selected_pair_order = tuple(
            (candidate.query_id, candidate.context_id)
            for candidate in universe.candidates
            if (candidate.query_id, candidate.context_id) in selected_pairs
        )
        stage_evidence = _filter_evidence(evidence, selected_pairs)
        stage_availability = evidence_availability.filter_pairs(selected_pair_order)
        access = StageChunkAccess(
            chunk_store,
            tuple(selected_pair_order),
        )
        stage_input_hash = _stage_input_identity(
            step,
            selected_pair_order,
            questions,
            stage_evidence,
            stage_availability,
            chunk_store,
            evidence_policy,
        )
        _trace(
            "stage.start",
            stage=step.stage_name,
            adapter_id=step.adapter_id,
            selected_pairs=len(selected_pair_order),
            evidence_records=len(stage_evidence.records),
            stage_input_hash=stage_input_hash,
        )
        stage_input = JudgeStageInput(
            stage=step,
            questions=questions,
            evidence=stage_evidence,
            evidence_availability=stage_availability,
            chunk_store=access,
            selected_pairs=selected_pair_order,
            stage_input_hash=stage_input_hash,
            provenance=_stage_provenance(step, descriptors[step.adapter_id]),
        )
        raw_key = ContentAddressedCache.raw_inference_key(
            stage_input,
            descriptors[step.adapter_id],
            runtime,
            producer_code_sha256=producer_code_sha256,
        )
        cached = cache.load_json("raw_inference", raw_key) if cache else None
        if cached is not None:
            if (
                cached.get("stage_input_hash") != stage_input.stage_input_hash
                or cached.get("corpus_sha256") != chunk_store.corpus_sha256
                or cached.get("producer_code_sha256") != producer_code_sha256
            ):
                raise ValueError("Raw inference cache stage input identity mismatch")
            consumed_chunk_ids = cached.get("consumed_chunk_ids", [])
            if not isinstance(consumed_chunk_ids, list):
                raise ValueError("Raw inference cache has invalid consumed_chunk_ids")
            consumed_chunk_pairs = cached.get("consumed_chunk_pairs", [])
            if not isinstance(consumed_chunk_pairs, list):
                raise ValueError("Raw inference cache has invalid consumed_chunk_pairs")
            try:
                cached_chunk_pairs = tuple(
                    (str(item["query_id"]), str(item["chunk_id"]))
                    for item in consumed_chunk_pairs
                )
            except (KeyError, TypeError) as exc:
                raise ValueError("Raw inference cache has malformed consumed_chunk_pairs") from exc
            output = JudgeStageOutput(
                signals=_ranking_signals_from_cache(cached),
                consumed_chunk_ids=tuple(str(item) for item in consumed_chunk_ids),
                consumed_chunk_pairs=cached_chunk_pairs,
            )
            validated_chunk_ids, validated_chunk_pairs = _validated_consumed_chunks(
                output, access, chunk_store, selected_pairs
            )
            output = replace(
                output,
                consumed_chunk_ids=validated_chunk_ids,
                consumed_chunk_pairs=validated_chunk_pairs,
            )
            cache_hit = True
        else:
            cuda_telemetry_probe = _reset_cuda_peak_memory(runtime)
            cuda_telemetry_unavailable_reason = (
                None
                if cuda_telemetry_probe is None
                else cuda_telemetry_probe.unavailable_reason
            )
            session = adapters[step.adapter_id].open(runtime)
            try:
                output = session.produce(stage_input)
            finally:
                session.close()
            cuda_peak_allocated_bytes, cuda_peak_reserved_bytes = _cuda_peak_memory_bytes(
                cuda_telemetry_probe
            )
            validated_chunk_ids, validated_chunk_pairs = _validated_consumed_chunks(
                output, access, chunk_store, selected_pairs
            )
            output = replace(
                output,
                consumed_chunk_ids=validated_chunk_ids,
                consumed_chunk_pairs=validated_chunk_pairs,
            )
            cache_hit = False
        if cache_hit:
            cuda_peak_allocated_bytes = None
            cuda_peak_reserved_bytes = None
            cuda_telemetry_unavailable_reason = None
        derived_signal_cache_key: str | None = None
        derived_signal_cache_hit: bool | None = None
        if step.projection is not None:
            projection = step.projection
            raw_signal_hash = semantic_sha256(output.signals)
            rank_tie_break_hash = _pointwise_rank_tie_break_hash(
                universe, ledger.snapshot(), projection, selected_pairs
            )
            derived_signal_cache_key = ContentAddressedCache.derived_signal_key(
                raw_signal_hash,
                projection,
                stage_input_hash=stage_input.stage_input_hash,
                rank_tie_break_hash=rank_tie_break_hash,
                producer_code_sha256=producer_code_sha256,
            )
            cached_derived = (
                cache.load_json("derived_signal", derived_signal_cache_key)
                if cache is not None
                else None
            )
            if cached_derived is not None:
                if (
                    cached_derived.get("stage_input_hash")
                    != stage_input.stage_input_hash
                    or cached_derived.get("corpus_sha256")
                    != chunk_store.corpus_sha256
                    or cached_derived.get("raw_signal_hash") != raw_signal_hash
                    or cached_derived.get("rank_tie_break_hash") != rank_tie_break_hash
                    or cached_derived.get("producer_code_sha256")
                    != producer_code_sha256
                    or semantic_sha256(cached_derived.get("projection"))
                    != semantic_sha256(projection)
                ):
                    raise ValueError("Derived-signal cache input identity mismatch")
                _validated_pointwise_raw_signals(
                    descriptors[step.adapter_id],
                    stage_input,
                    output,
                    selected_pairs,
                )
                stage_signals = _ranking_signals_from_cache(
                    cached_derived, schema="derived-ranking-signals-v1"
                )
                if (
                    cached_derived.get("output_signal_hash")
                    != semantic_sha256(stage_signals)
                ):
                    raise ValueError("Derived-signal cache content hash mismatch")
                derived_signal_cache_hit = True
            else:
                stage_signals = _derive_pointwise_signals(
                    universe,
                    ledger.snapshot(),
                    descriptors[step.adapter_id],
                    stage_input,
                    output,
                    selected_pairs,
                )
                derived_signal_cache_hit = False
                if cache is not None:
                    cache.store_json(
                        "derived_signal",
                        derived_signal_cache_key,
                        {
                            "schema": "derived-ranking-signals-v1",
                            "stage_input_hash": stage_input.stage_input_hash,
                            "corpus_sha256": chunk_store.corpus_sha256,
                            "raw_signal_hash": raw_signal_hash,
                            "projection": _semantic_value(projection),
                            "rank_tie_break_hash": rank_tie_break_hash,
                            "producer_code_sha256": producer_code_sha256,
                            "output_signal_hash": semantic_sha256(stage_signals),
                            "signals": [
                                _signal_artifact_row(signal)
                                for signal in stage_signals
                            ],
                        },
                    )
        else:
            stage_signals = _validate_direct_signals(
                step,
                descriptors[step.adapter_id],
                stage_input,
                output,
                selected_pairs,
            )
        if cache is not None and not cache_hit:
            cache.store_json(
                "raw_inference",
                raw_key,
                {
                    "schema": "raw-ranking-signals-v3",
                    "stage_input_hash": stage_input.stage_input_hash,
                    "corpus_sha256": chunk_store.corpus_sha256,
                    "producer_code_sha256": producer_code_sha256,
                    "signals": [
                        _signal_artifact_row(signal) for signal in output.signals
                    ],
                    "consumed_chunk_ids": list(output.consumed_chunk_ids),
                    "consumed_chunk_pairs": [
                        {"query_id": query_id, "chunk_id": chunk_id}
                        for query_id, chunk_id in output.consumed_chunk_pairs
                    ],
                },
            )
        ledger.append_batch(stage_signals)
        execution_records.append(
            StageExecutionRecord(
                stage_name=step.stage_name,
                input_view=step.input_view,
                stage_input_hash=stage_input.stage_input_hash,
                corpus_sha256=chunk_store.corpus_sha256,
                evidence_hash=stage_input.evidence.content_hash,
                evidence_availability_hash=stage_availability.content_hash,
                adapter_config_hash=stage_input.provenance.config_hash,
                raw_cache_key=raw_key,
                cache_hit=cache_hit,
                derived_signal_cache_key=derived_signal_cache_key,
                derived_signal_cache_hit=derived_signal_cache_hit,
                consumed_chunk_ids=output.consumed_chunk_ids,
                consumed_chunk_pairs=output.consumed_chunk_pairs,
                output_signal_hash=semantic_sha256(stage_signals),
                cuda_peak_allocated_bytes=cuda_peak_allocated_bytes,
                cuda_peak_reserved_bytes=cuda_peak_reserved_bytes,
                cuda_telemetry_unavailable_reason=cuda_telemetry_unavailable_reason,
            )
        )
        _trace(
            "stage.complete",
            stage=step.stage_name,
            cache_hit=cache_hit,
            derived_signal_cache_hit=derived_signal_cache_hit,
            output_signals=len(stage_signals),
        )
        return
    if isinstance(step, Select):
        views[step.stage_name] = _select_view(universe, ledger, step, views)
        return
    if isinstance(step, Cascade):
        for child in step.steps:
            _execute_recipe_step(
                child,
                universe,
                ledger,
                views,
                questions,
                evidence,
                evidence_availability,
                chunk_store,
                evidence_policy,
                adapters,
                descriptors,
                runtime,
                cache,
                producer_code_sha256,
                execution_records,
            )
        return
    if isinstance(step, Parallel):
        base_snapshot = ledger.snapshot()
        pending_signals: list[RankingSignal] = []
        pending_views: dict[str, ShortlistView] = {}
        for branch in step.branches:
            branch_ledger = TypedSignalLedger(universe)
            branch_ledger.append_batch(base_snapshot.signals)
            branch_views = dict(views)
            _execute_recipe_step(
                branch,
                universe,
                branch_ledger,
                branch_views,
                questions,
                evidence,
                evidence_availability,
                chunk_store,
                evidence_policy,
                adapters,
                descriptors,
                runtime,
                cache,
                producer_code_sha256,
                execution_records,
            )
            pending_signals.extend(branch_ledger.snapshot().signals[len(base_snapshot.signals) :])
            pending_views.update(
                {
                    view_id: view
                    for view_id, view in branch_views.items()
                    if view_id not in views
                }
            )
        ledger.append_batch(pending_signals)
        views.update(pending_views)
        return
    raise TypeError(f"Unsupported RecipeStep: {type(step).__name__}")

def _signal_artifact_row(signal: RankingSignal) -> dict[str, Any]:
    return {
        "kind": type(signal).__name__,
        "signal_id": _signal_id(signal),
        **_semantic_value(signal),
    }

def execute_validated_rerank_request(
    request: RerankRequest,
    adapter_registry: Mapping[str, JudgeAdapter],
) -> RerankRunResult:
    _trace(
        "request.start",
        run_id=request.run_identity.run_id,
        split=request.run_identity.evaluation_split,
        host_account=request.run_identity.host_account,
        queries=len(request.questions),
        candidates=len(request.candidate_universe.candidates),
        device=request.runtime.device,
        batch_size=request.runtime.batch_size,
        max_length=request.runtime.max_length,
        bf16=request.runtime.bf16,
    )
    if set(request.questions) != set(request.candidate_universe.query_order):
        raise ValueError("Query IDs differ between questions and Candidate Universe")
    if (
        request.complete_system_budget is not None
        and request.complete_system_budget.status is BudgetStatus.EXCEEDS_LIMIT
    ):
        raise ValueError(
            "Recipe exceeds the complete-system 4B budget: "
            f"{request.complete_system_budget.projected_parameter_count}"
        )
    producer_code_sha256 = request.producer_code_sha256 or sha256_file(__file__)
    ledger = TypedSignalLedger(request.candidate_universe)
    ledger.append_batch(request.initial_signals)
    descriptors = {adapter_id: adapter.describe() for adapter_id, adapter in adapter_registry.items()}
    compiled = RecipeCompiler.compile(
        recipe=request.recipe,
        universe=request.candidate_universe,
        ledger=ledger.snapshot(),
        adapter_descriptors=descriptors,
        finalizer_requirements=_finalizer_requirements(request.finalizer),
    )
    evidence_availability = EvidenceResolver.collect_availability(
        request.candidate_universe,
        request.fused_hits,
        request.branch_hits,
        request.evidence_policy,
    )
    evidence_cache_key: str | None = None
    evidence_cache_hit: bool | None = None
    if request.cache is not None:
        evidence_cache_key = ContentAddressedCache.evidence_key(
            request.candidate_universe,
            request.chunk_store,
            request.evidence_policy,
            evidence_availability,
            producer_code_sha256=producer_code_sha256,
        )
        cached_evidence = request.cache.load_json("evidence", evidence_cache_key)
    else:
        cached_evidence = None
    if cached_evidence is not None:
        evidence = _evidence_batch_from_cache(
            cached_evidence,
            request.candidate_universe,
            request.chunk_store,
            request.evidence_policy,
            evidence_availability,
            producer_code_sha256,
        )
        evidence_cache_hit = True
    else:
        evidence = EvidenceResolver.resolve(
            universe=request.candidate_universe,
            fused_hits=request.fused_hits,
            branch_hits=request.branch_hits,
            store=request.chunk_store,
            policy=request.evidence_policy,
            availability=evidence_availability,
            questions=request.questions,
        )
        evidence_cache_hit = False if request.cache is not None else None
        if request.cache is not None and evidence_cache_key is not None:
            request.cache.store_json(
                "evidence",
                evidence_cache_key,
                _evidence_cache_payload(
                    evidence,
                    request.candidate_universe,
                    request.chunk_store,
                    request.evidence_policy,
                    evidence_availability,
                    producer_code_sha256,
                ),
            )
    _trace(
        "evidence.ready",
        cache_hit=evidence_cache_hit,
        records=len(evidence.records),
        evidence_hash=evidence.content_hash,
    )
    views: dict[str, ShortlistView] = {}
    stage_executions: list[StageExecutionRecord] = []
    _execute_recipe_step(
        compiled.recipe,
        request.candidate_universe,
        ledger,
        views,
        request.questions,
        evidence,
        evidence_availability,
        request.chunk_store,
        request.evidence_policy,
        adapter_registry,
        descriptors,
        request.runtime,
        request.cache,
        producer_code_sha256,
        stage_executions,
    )
    _trace(
        "inference.complete",
        stages=len(stage_executions),
        ledger_signals=len(ledger.snapshot().signals),
    )
    snapshot = ledger.snapshot()
    finalizer_key = ContentAddressedCache.finalizer_key(
        request.candidate_universe,
        snapshot,
        request.finalizer,
        producer_code_sha256=producer_code_sha256,
    )
    cached_finalizer = (
        request.cache.load_json("finalizer", finalizer_key) if request.cache else None
    )
    if cached_finalizer is not None:
        if cached_finalizer.get("producer_code_sha256") != producer_code_sha256:
            raise ValueError("Finalizer cache producer code identity mismatch")
        final_ranking = _final_ranking_from_cache(cached_finalizer)
    else:
        final_ranking = request.finalizer.finalize(request.candidate_universe, snapshot)
        if request.cache:
            request.cache.store_json(
                "finalizer",
                finalizer_key,
                {
                    "schema": "final-ranking-v1",
                    "producer_code_sha256": producer_code_sha256,
                    "rows": [_semantic_value(row) for row in final_ranking.rows],
                },
            )
    submission = {
        query_id: {"answer": list(final_ranking.context_ids(query_id)[:5])}
        for query_id in request.candidate_universe.query_order
    }
    writer = ArtifactWriter(request.output_dir)
    try:


        writer.write_json(
            "candidate_universe.json",
            {
                query_id: list(request.candidate_universe.context_ids(query_id))
                for query_id in request.candidate_universe.query_order
            },
        )


        writer.write_json(
            "candidate_universe_order.json",
            {
                "schema_version": "candidate-universe-order-v1",
                "query_order": list(request.candidate_universe.query_order),
                "content_hash": request.candidate_universe.content_hash,
            },
        )
        writer.write_jsonl("evidence.jsonl", evidence.records)
        writer.write_jsonl("stage_executions.jsonl", stage_executions)
        writer.write_jsonl("final_ranking.jsonl", final_ranking.rows)
        writer.write_json("submission.json", submission)
        by_channel: dict[str, list[RankingSignal]] = defaultdict(list)
        for signal in snapshot.signals:
            by_channel[signal.channel].append(signal)
        for channel, signals in by_channel.items():
            writer.write_jsonl(
                f"signals/{channel}.jsonl",
                tuple(_signal_artifact_row(signal) for signal in signals),
            )
        writer.write_json(
            "run_manifest.json",
            {
                "pipeline_stage": request.run_identity.pipeline_stage,
                "host_account": request.run_identity.host_account,
                "run_id": request.run_identity.run_id,
                "evaluation_split": request.run_identity.evaluation_split,
                "workflow": os.environ.get("RERANK_WORKFLOW"),
                "workflow_config_sha256": os.environ.get("RERANK_WORKFLOW_CONFIG_SHA256"),
                "candidate_universe_hash": request.candidate_universe.content_hash,
                "evidence_availability_hash": evidence_availability.content_hash,
                "evidence_hash": evidence.content_hash,
                "evidence_cache_key": evidence_cache_key,
                "evidence_cache_hit": evidence_cache_hit,
                "ledger_hash": snapshot.content_hash,
                "recipe_hash": compiled.recipe_hash,
                "finalizer_hash": semantic_sha256(request.finalizer),
                "corpus_sha256": request.chunk_store.corpus_sha256,
                "script": str(Path(__file__).resolve()),
                "script_sha256": producer_code_sha256,
                "launcher": {
                    "path": os.environ.get("RERANK_LAUNCHER_PATH"),
                    "sha256": os.environ.get("RERANK_LAUNCHER_SHA256"),
                },
                "model": {
                    "adapter_id": _recipe_judges(compiled.recipe)[0].adapter_id,
                    "descriptor": _semantic_value(
                        descriptors[_recipe_judges(compiled.recipe)[0].adapter_id]
                    ),
                },
                "models": {
                    judge.adapter_id: {
                        "descriptor": _semantic_value(descriptors[judge.adapter_id]),
                    }
                    for judge in _recipe_judges(compiled.recipe)
                },
                "runtime_closure": (
                    _semantic_value(request.runtime_closure)
                    if request.runtime_closure is not None
                    else None
                ),
                "runtime": _semantic_value(request.runtime),
                "runtime_provenance": _runtime_provenance(request.runtime),
                "whole_system_parameter_budget": (
                    asdict(request.complete_system_budget)
                    if request.complete_system_budget is not None
                    else None
                ),
                "evidence_policy": _semantic_value(request.evidence_policy),
                "input_provenance": _semantic_value(request.input_provenance),
                "output_dir": str(request.output_dir.resolve()),
                "created_at_utc": datetime.now(UTC).isoformat(),
            },
        )
        output_dir = writer.commit(
            {
                "status": "complete",
                "run_id": request.run_identity.run_id,
                "ledger_hash": snapshot.content_hash,
            }
        )
    except Exception:
        writer.abort()
        raise
    return RerankRunResult(
        output_dir=output_dir,
        final_ranking=final_ranking,
        evidence=evidence,
        ledger=snapshot,
        stage_executions=tuple(stage_executions),
    )

class RerankModule:

    def __init__(self, adapter_registry: Mapping[str, JudgeAdapter]) -> None:
        self._adapter_registry = dict(adapter_registry)

    def run(self, request: RerankRequest) -> RerankRunResult:
        return execute_validated_rerank_request(request, self._adapter_registry)

class SequenceClassificationPointwiseAdapter:

    def __init__(self, model_name: str, descriptor: JudgeDescriptor) -> None:
        if "score" not in descriptor.output_kinds:
            raise ValueError("Sequence adapter descriptor must declare score output")
        self._model_name = model_name
        self._descriptor = descriptor

    def describe(self) -> JudgeDescriptor:
        return self._descriptor

    def open(self, runtime: JudgeRuntime) -> "_SequenceClassificationPointwiseSession":
        if not runtime.device.startswith("cuda"):
            raise RuntimeError(
                "SequenceClassificationPointwiseAdapter requires CUDA; "
                "CPU fallback is disabled"
            )
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("torch and transformers are required for pointwise reranking") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but unavailable: {runtime.device}")
        if runtime.bf16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 requested but the selected CUDA device does not support BF16")
        dtype = torch.bfloat16 if runtime.bf16 else torch.float32
        _trace(
            "model.open.start",
            adapter_id=self._descriptor.adapter_id,
            model_dir=str(Path(self._model_name).resolve()),
            model_class=self._descriptor.model_class,
            parameter_count=self._descriptor.parameter_count,
            device=runtime.device,
            dtype=str(dtype),
            max_length=runtime.max_length,
            batch_size=runtime.batch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(self._model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self._model_name,
            torch_dtype=dtype,
            attn_implementation=runtime.attn_implementation,
        ).to(runtime.device)
        num_labels = int(getattr(model.config, "num_labels", 0))
        if num_labels != 1:
            del model
            del tokenizer
            raise RuntimeError(
                "Expected a one-logit sequence-classification reranker; "
                f"model reports num_labels={num_labels}"
            )
        model.eval()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        _trace(
            "model.open.complete",
            adapter_id=self._descriptor.adapter_id,
            num_labels=num_labels,
        )
        return _SequenceClassificationPointwiseSession(
            torch_module=torch,
            model=model,
            tokenizer=tokenizer,
            runtime=runtime,
        )

class _SequenceClassificationPointwiseSession:
    def __init__(self, torch_module: Any, model: Any, tokenizer: Any, runtime: JudgeRuntime) -> None:
        self._torch = torch_module
        self._model = model
        self._tokenizer = tokenizer
        self._runtime = runtime
        self._closed = False

    def produce(self, stage_input: JudgeStageInput) -> JudgeStageOutput:
        if self._closed:
            raise RuntimeError("Pointwise session is closed")
        values: list[float] = []
        ordered_evidence = order_pointwise_evidence_by_length(
            stage_input.evidence, stage_input.questions
        )
        records = ordered_evidence.records
        total_batches = (len(records) + self._runtime.batch_size - 1) // self._runtime.batch_size
        print(
            "[RERANK-PROGRESS] "
            f"phase=pointwise_inference status=start records={len(records)} "
            f"batches={total_batches}",
            flush=True,
        )
        for batch_number, start in enumerate(
            range(0, len(records), self._runtime.batch_size), 1
        ):
            batch = records[start : start + self._runtime.batch_size]
            pairs = [
                [stage_input.questions[evidence.query_id], evidence.text]
                for evidence in batch
            ]
            inputs = self._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=self._runtime.max_length,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self._runtime.device) for key, value in inputs.items()
            }
            with self._torch.inference_mode():
                logits = self._model(**inputs, return_dict=True).logits
            if logits.ndim != 2 or logits.shape[1] != 1:
                raise RuntimeError(
                    f"Pointwise model produced invalid logits shape={tuple(logits.shape)}"
                )
            values.extend(float(item) for item in logits[:, 0].float().cpu().tolist())
            if batch_number == total_batches or batch_number % 512 == 0:
                print(
                    "[RERANK-PROGRESS] "
                    f"phase=pointwise_inference status=running batch={batch_number}/"
                    f"{total_batches} scored={len(values)}",
                    flush=True,
                )
        ordered_input = replace(stage_input, evidence=ordered_evidence)
        output = JudgeStageOutput.chunk_score_values(ordered_input, values)
        print(
            "[RERANK-PROGRESS] "
            f"phase=pointwise_inference status=complete scored={len(values)}",
            flush=True,
        )
        return output

    def close(self) -> None:
        if self._closed:
            return
        device = self._runtime.device
        del self._model
        del self._tokenizer
        if device.startswith("cuda"):
            self._torch.cuda.synchronize(device)
            self._torch.cuda.empty_cache()
        self._closed = True


_load_candidate_inputs = load_candidate_inputs


PARAMETER_BUDGET = 4_000_000_000

SYSTEM_INVENTORY_SCHEMA = "legalir-system-model-inventory-v1"

class BudgetStatus(str, Enum):
    WITHIN_LIMIT = "within_limit"
    EXCEEDS_LIMIT = "exceeds_limit"

@dataclass(frozen=True)
class ProjectedSystemBudget:
    candidate_parameter_count: int
    projected_parameter_count: int
    status: BudgetStatus
    overage: int


def load_system_inventory_report(path: Path) -> tuple[int, bool]:
    payload = _read_json_mapping(path)
    if payload.get("schema_version") != SYSTEM_INVENTORY_SCHEMA:
        raise ValueError(f"Unsupported system inventory: {path}")
    components = payload.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("System inventory has no components")
    total = 0
    bound = True
    for component in components:
        if not isinstance(component, Mapping):
            raise ValueError("System inventory component is not a mapping")
        count = component.get("parameter_count")
        name = component.get("name")
        if (
            not isinstance(name, str)
            or not name
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("System inventory component lacks a valid count")
        total += count
        if component.get("kind") != "sparse":
            checkpoint = component.get("checkpoint_sha256")
            if not isinstance(checkpoint, str) or not checkpoint:
                bound = False
    verified_total = payload.get("verified_total")
    if not isinstance(verified_total, int) or verified_total != total:
        raise ValueError("System inventory total does not match its components")
    return total, bound


def complete_system_budget(
    inventory_path: Path, descriptors: Sequence[JudgeDescriptor]
) -> ProjectedSystemBudget:
    declared_total, inventory_fully_bound = load_system_inventory_report(inventory_path)
    if not inventory_fully_bound:
        raise ValueError(
            "Complete-system budget requires checkpoint_sha256 for every non-sparse inventory component"
        )
    payload = _read_json_mapping(inventory_path)
    components = payload["components"]
    assert isinstance(components, list)
    inventory_checkpoints = {
        str(component["checkpoint_sha256"])
        for component in components
        if isinstance(component, Mapping)
        and component.get("kind") != "sparse"
        and isinstance(component.get("checkpoint_sha256"), str)
        and component["checkpoint_sha256"]
    }
    recipe_models = {
        descriptor.checkpoint_sha256: descriptor
        for descriptor in descriptors
    }
    new_parameters = sum(
        descriptor.parameter_count
        for checkpoint, descriptor in recipe_models.items()
        if checkpoint not in inventory_checkpoints
    )
    return projected_system_budget(new_parameters, system_total=declared_total)


def projected_system_budget(
    candidate_parameter_count: int, *, system_total: int
) -> ProjectedSystemBudget:
    if candidate_parameter_count < 0:
        raise ValueError("candidate parameter count must be non-negative")
    if system_total < 0:
        raise ValueError("system total must be non-negative")
    projected = system_total + candidate_parameter_count
    overage = max(0, projected - PARAMETER_BUDGET)
    return ProjectedSystemBudget(
        candidate_parameter_count=candidate_parameter_count,
        projected_parameter_count=projected,
        status=(
            BudgetStatus.WITHIN_LIMIT
            if overage == 0
            else BudgetStatus.EXCEEDS_LIMIT
        ),
        overage=overage,
    )


def _runtime_provenance(runtime: JudgeRuntime) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "runtime": asdict(runtime),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "seed": 0,
        "deterministic_algorithms": False,
    }
    try:
        import torch
        import transformers

        payload.update(
            {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": (
                    torch.cuda.get_device_name(runtime.device)
                    if runtime.device.startswith("cuda") and torch.cuda.is_available()
                    else None
                ),
                "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            }
        )
    except ImportError:
        payload["torch"] = None
        payload["transformers"] = None
    return payload


def _require_paths(args: argparse.Namespace) -> None:
    for path in (args.queries, args.corpus, args.fused_ranked):
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)
    for path in [args.bm25_ranked, *args.dense_ranked]:
        if path is not None and not path.is_file():
            raise FileNotFoundError(path)

def _adapter_registry_from_manifest(
    args: argparse.Namespace,
    adapter_registry: Mapping[str, JudgeAdapter] | None,
) -> tuple[Mapping[str, JudgeAdapter], dict[str, Any] | None]:
    if adapter_registry is not None:
        if args.adapter_id not in adapter_registry:
            raise ValueError(f"Injected adapter registry lacks {args.adapter_id}")
        return adapter_registry, None
    manifest_path = args.verify_runtime_manifest or args.model_manifest
    if manifest_path is None or not manifest_path.is_file():
        raise ValueError(
            "Production run requires --model-manifest "
            "(or --verify-runtime-manifest for a pinned closure)"
        )
    descriptor, model_dir, report = load_verified_judge_model(manifest_path)
    if descriptor.adapter_id != args.adapter_id:
        raise ValueError("--adapter-id differs from model manifest")
    closure = {**report, "verifier": "rerank_typed._adapter_registry_from_manifest"}
    return {
        args.adapter_id: SequenceClassificationPointwiseAdapter(
            model_name=str(model_dir), descriptor=descriptor
        )
    }, closure

def _build_request(
    args: argparse.Namespace,
    adapter_registry: Mapping[str, JudgeAdapter],
    runtime_closure: Mapping[str, Any] | None,
) -> RerankRequest:
    _require_paths(args)
    questions, universe, initial_signals, fused_hits, branches = _load_candidate_inputs(args)
    descriptor = adapter_registry[args.adapter_id].describe()
    chunk_store = CanonicalChunkStore(args.corpus)
    producer_code_sha256, _ = source_closure_sha256()
    input_provenance = {
        "queries": _artifact_provenance(args.queries),
        "corpus": _artifact_provenance(args.corpus),
        "fused_ranked": _artifact_provenance(args.fused_ranked),
        **{
            f"dense_ranked_{index}": _artifact_provenance(path)
            for index, path in enumerate(args.dense_ranked)
        },
    }
    policy = EvidencePolicy.champion_v1(
        max_chunks=args.max_chunks_per_context,
        use_article_view=args.use_article_view,
        neighbor_context=args.neighbor_context,
        question_order_min_gain=args.question_order_min_gain,
    )
    projection = PointwiseProjection(
        raw_chunk_channel="ce_raw_logit",
        context_score_channel="ce_context_max",
        rank_channel="ce_rank",
        rank_tie_break_channel="fusion_rank",
        score_scale="raw_logit",
    )
    if not 0.0 <= args.original_weight <= 1.0 or args.rrf_k < 0:
        raise ValueError("Invalid RRF settings")
    finalizer = RrfFinalizer(
        channels=(
            RankChannel("fusion_rank", args.original_weight),
            RankChannel("ce_rank", 1.0 - args.original_weight),
        ),
        rrf_k=args.rrf_k,
        tie_break_channels=("ce_rank",),
    )
    split = "private"
    recipe = Judge(
        stage_name="pointwise_cross_encoder",
        adapter_id=args.adapter_id,
        input_view="universe",
        output_channels=projection.output_channels,
        projection=projection,
    )
    common = {
        "run_identity": RunIdentity(
            run_id=f"{args.mode}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
            pipeline_stage="rerank",
            evaluation_split=split,
            host_account=str(_runtime_roots()["host_account"]),
        ),
        "questions": questions,
        "candidate_universe": universe,
        "initial_signals": initial_signals,
        "chunk_store": chunk_store,
        "fused_hits": fused_hits,
        "branch_hits": branches,
        "evidence_policy": policy,
        "recipe": recipe,
        "finalizer": finalizer,
        "output_dir": args.output_dir,
        "runtime": JudgeRuntime(
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            bf16=args.bf16,
            attn_implementation=args.attn_implementation,
        ),
        "cache": ContentAddressedCache(args.cache_dir) if args.cache_dir else None,
        "input_provenance": input_provenance,
        "producer_code_sha256": producer_code_sha256,
        "runtime_closure": runtime_closure,
        "complete_system_budget": (
            complete_system_budget(
                inventory_path,
                tuple(
                    adapter.describe() for adapter in adapter_registry.values()
                ),
            )
            if (inventory_path := _env_path("SYSTEM_INVENTORY")) is not None
            else None
        ),
    }
    return RerankRequest(**common)


@dataclass(frozen=True)
class WorkflowSpec:
    name: str
    split: str | None
    requires_gpu: bool


WORKFLOW_SPECS: dict[str, WorkflowSpec] = {
    "private": WorkflowSpec("private", "private", True),
}


def _env(name: str, default: str | None = None, *, required: bool = False) -> str:
    value = os.environ.get(name, default)
    if value is None or (required and not value.strip()):
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _env_path(name: str, default: str | Path | None = None, *, required: bool = False) -> Path | None:
    raw_default = None if default is None else str(default)
    value = os.environ.get(name, raw_default)
    if value is None or value == "":
        if required:
            raise ValueError(f"Missing required environment variable: {name}")
        return None
    return Path(value)


def _env_path_list(name: str, default: Sequence[Path] = ()) -> list[Path]:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return [Path(item) for item in default]

    return [Path(item) for item in raw.split(":") if item]


def _runtime_roots() -> dict[str, Path | str]:
    project_root = Path(_env("PROJECT_ROOT", "/path/to/project"))
    retrieval_dir = Path(_env("RETRIEVAL_DIR", str(project_root / "retrieve/local")))
    query_workspace = Path(
        _env(
            "QUERY_WORKSPACE",
            str(project_root / "preprocess/workspace/preprocess_compact_standard"),
        )
    )
    return {
        "project_root": project_root,
        "retrieval_dir": retrieval_dir,
        "code_dir": Path(_env("CODE_DIR", str(retrieval_dir))),
        "query_workspace": query_workspace,
        "host_account": _env("HOST_ACCOUNT", "host"),
    }


def _job_suffix() -> str:
    return os.environ.get("SLURM_JOB_ID") or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _default_split_inputs(split: str) -> dict[str, Any]:
    roots = _runtime_roots()
    retrieval_dir = Path(roots["retrieval_dir"])
    workspace = Path(roots["query_workspace"])
    queries = _env_path("QUERIES", workspace / "queries" / f"{split}.jsonl", required=True)
    corpus = _env_path("CORPUS", workspace / "corpus/retrieval_chunks.jsonl", required=True)

    fused = _env_path(
        "FUSED_RANKED",
        retrieval_dir / "outputs" / "fusion" / "fused_private" / "ranked.jsonl",
        required=True,
    )
    bm25 = _env_path(
        "BM25_RANKED",
        retrieval_dir / "outputs" / f"bm25_{split}" / "ranked.jsonl",
        required=True,
    )
    dense_defaults = (
        retrieval_dir / "outputs" / f"bgem3_{split}" / "ranked.jsonl",
        retrieval_dir / "outputs" / f"vae_{split}" / "ranked.jsonl",
        retrieval_dir / "outputs" / f"f2llm_{split}" / "ranked.jsonl",
    )
    dense = _env_path_list("DENSE_RANKED", dense_defaults)
    return {
        "queries": queries,
        "corpus": corpus,
        "fused_ranked": fused,
        "bm25_ranked": bm25,
        "dense_ranked": dense,
    }


def _workflow_config_hash(workflow: str, payload: Mapping[str, Any]) -> str:
    return semantic_sha256({"workflow": workflow, "config": payload})


def _log_dispatch(workflow: str, config_payload: Mapping[str, Any]) -> None:
    producer_hash, source_files = source_closure_sha256()
    message = {
        "pipeline_stage": "rerank",
        "workflow": workflow,
        "host_account": _runtime_roots()["host_account"],
        "producer_code_sha256": producer_hash,
        "producer_files": source_files,
        "workflow_config_sha256": _workflow_config_hash(workflow, config_payload),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": config_payload,
    }
    print("[RERANK-DISPATCH] " + json.dumps(message, ensure_ascii=False, sort_keys=True, default=str))


def _private_namespace() -> argparse.Namespace:
    inputs = _default_split_inputs("private")
    roots = _runtime_roots()
    retrieval_dir = Path(roots["retrieval_dir"])
    output_default = retrieval_dir / "outputs" / f"rerank_private_{_job_suffix()}"
    output_dir = _env_path("OUTPUT_DIR", output_default, required=True)
    cache_dir = _env_path("CACHE_DIR")
    model_manifest = _env_path(
        "CHAMPION_MANIFEST",
        retrieval_dir / "rerank_champion_model_manifest.json",
        required=True,
    )
    runtime_manifest = _env_path("CHAMPION_RUNTIME_MANIFEST")
    manifest_to_load = runtime_manifest or model_manifest
    if manifest_to_load is None or not manifest_to_load.is_file():
        raise FileNotFoundError(manifest_to_load)
    descriptor, _, _ = load_verified_judge_model(manifest_to_load)
    return argparse.Namespace(
        mode="champion-v1",
        queries=inputs["queries"],
        corpus=inputs["corpus"],
        fused_ranked=inputs["fused_ranked"],
        bm25_ranked=inputs["bm25_ranked"],
        dense_ranked=inputs["dense_ranked"],
        output_dir=output_dir,
        adapter_id=descriptor.adapter_id,
        model_manifest=model_manifest,
        verify_runtime_manifest=runtime_manifest,
        cache_dir=cache_dir,
        device=_env("DEVICE", "cuda:0"),
        batch_size=8,
        max_length=2304,
        bf16=True,
        attn_implementation="sdpa",
        candidate_depth=200,
        max_chunks_per_context=2,
        question_order_min_gain=0,
        use_article_view=True,
        neighbor_context=0,
        original_weight=0.5,
        rrf_k=20,
        system_inventory=None,
    )


def _run_typed_namespace(args: argparse.Namespace) -> RerankRunResult:
    registry, runtime_closure = _adapter_registry_from_manifest(args, None)
    request = _build_request(args, registry, runtime_closure)
    return RerankModule(adapter_registry=registry).run(request)


def workflow_private() -> Path:
    args = _private_namespace()
    return _run_typed_namespace(args).output_dir


WORKFLOW_HANDLERS: Mapping[str, Callable[[], Path]] = {
    "private": workflow_private,
}


def _dispatch_config_payload(workflow: str) -> dict[str, Any]:
    roots = _runtime_roots()
    keys = (
        "PROJECT_ROOT", "RETRIEVAL_DIR", "CODE_DIR", "QUERY_WORKSPACE", "HOST_ACCOUNT",
        "QUERIES", "CORPUS", "FUSED_RANKED", "BM25_RANKED", "DENSE_RANKED",
        "OUTPUT_DIR", "CACHE_DIR", "CHAMPION_MANIFEST", "CHAMPION_RUNTIME_MANIFEST",
        "DEVICE", "SYSTEM_INVENTORY",
    )
    return {
        "workflow": workflow,
        "roots": {key: str(value) for key, value in roots.items()},
        "environment": {key: os.environ[key] for key in keys if key in os.environ},
    }


def _validate_workflow_environment(spec: WorkflowSpec) -> None:
    if spec.requires_gpu and _env("DEVICE", "cuda:0").startswith("cpu"):
        raise ValueError(f"Workflow {spec.name} requires a GPU DEVICE")


def main() -> int:
    try:
        workflow = os.environ.get("RERANK_WORKFLOW", "").strip()
        if not workflow:
            raise ValueError(
                f"RERANK_WORKFLOW is required; allowed={sorted(WORKFLOW_SPECS)}"
            )
        spec = WORKFLOW_SPECS.get(workflow)
        handler = WORKFLOW_HANDLERS.get(workflow)
        if spec is None or handler is None:
            raise ValueError(
                f"Unknown RERANK_WORKFLOW={workflow!r}; allowed={sorted(WORKFLOW_HANDLERS)}"
            )
        _validate_workflow_environment(spec)
        config_payload = _dispatch_config_payload(workflow)
        os.environ["RERANK_WORKFLOW_CONFIG_SHA256"] = _workflow_config_hash(
            workflow, config_payload
        )
        _log_dispatch(workflow, config_payload)
        output = handler()
        print(json.dumps({"status": "complete", "workflow": workflow, "output": str(output)}, ensure_ascii=False))
        return 0
    except KeyboardInterrupt:
        print("ERROR: interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(
            "[RERANK-FAILURE] "
            + json.dumps(
                {
                    "workflow": os.environ.get("RERANK_WORKFLOW"),
                    "host_account": os.environ.get("HOST_ACCOUNT"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "node": os.environ.get("SLURM_NODELIST"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
