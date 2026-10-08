"""Hàm dùng chung cho các bước trong pipeline.

File này làm gì
----------------
Chứa các hàm mà nhiều bước cùng dùng: đọc ghi file JSON và JSONL an toàn, chuẩn hoá văn
bản tiếng Việt, tách từ pháp lý, băm sha256, đếm tham số trong file checkpoint, dựng và
kiểm tra submission, ghi index theo lô.

Pipeline stage
--------------
Không thuộc bước cụ thể nào. Được import bởi các bước khác.

Input / Output
--------------
Xem docstring của từng hàm. Phần lớn nhận `Path` và trả về dict hoặc iterator.

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
import os
import re
import shutil
import tempfile
import unicodedata
import uuid
from array import array
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


INDEX_FORMAT_VERSION = 1
RETRIEVAL_FIELDS = {"chunk_id", "context_id", "text"}
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
MULTISPACE_RE = re.compile(r"[^\S\n]+")
TAG_RE = re.compile(r"^\[([^\]]+)]\s*(.*)$")
WORD_RE = re.compile(r"[^\W_]+(?:_[^\W_]+)*", flags=re.UNICODE)
UNICODE_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2015\u2212\ufe58\ufe63\uff0d"
CODE_PUNCT_TRANSLATION = str.maketrans(
    {
        **{character: "-" for character in UNICODE_DASHES},
        "／": "/",
        "⁄": "/",
    }
)


DOCUMENT_NUMBER_RE = re.compile(
    r"(?i)(?<![\w])(?:"
    r"TCVN\s+(?:ISO(?:/IEC)?\s+)?\d+(?:-\d+)*(?:\s*:\s*\d{4})?"
    rf"|QCVN\s*\d{{1,3}}(?:-\d+)?\s*:\s*\d{{2,4}}(?:\s*/\s*[0-9A-ZÀ-ỸĐ]+(?:\s*[-.{UNICODE_DASHES}]\s*[0-9A-ZÀ-ỸĐ]+)*)?"
    rf"|\d{{1,4}}\s*/\s*\d{{4}}\s*/\s*[0-9A-ZÀ-ỸĐ]+(?:\s*[-.{UNICODE_DASHES}]\s*[0-9A-ZÀ-ỸĐ]+)*"
    r")"
)
LEGAL_REFERENCE_RE = re.compile(
    r"(?i)\b(điều|khoản|điểm|chương|mục|phụ\s+lục)\s+"
    r"(\d+[a-zđ]?|[ivxlcdm]+)(?:\s*[.)])?"
)


VIETNAMESE_QUERY_STOPWORDS = frozenset(
    {
        "ai",
        "bao",
        "bằng",
        "bị",
        "các",
        "cái",
        "cho",
        "có",
        "của",
        "đã",
        "đang",
        "đâu",
        "đây",
        "đó",
        "được",
        "gì",
        "hay",
        "khi",
        "là",
        "làm",
        "mà",
        "một",
        "nào",
        "này",
        "như",
        "những",
        "phải",
        "ra",
        "sao",
        "sẽ",
        "theo",
        "thì",
        "thế",
        "trong",
        "trên",
        "từ",
        "và",
        "về",
        "việc",
        "với",
    }
)

LEGAL_ABBREVIATIONS: dict[str, tuple[str, ...]] = {
    "bhxh": ("bảo", "hiểm", "xã", "hội"),
    "bhyt": ("bảo", "hiểm", "y", "tế"),
    "bhtn": ("bảo", "hiểm", "thất", "nghiệp"),
    "hđlđ": ("hợp", "đồng", "lao", "động"),
    "nld": ("người", "lao", "động"),
    "nlđ": ("người", "lao", "động"),
    "nsdld": ("người", "sử", "dụng", "lao", "động"),
    "nsdlđ": ("người", "sử", "dụng", "lao", "động"),
    "tnlđ": ("tai", "nạn", "lao", "động"),
    "pccc": ("phòng", "cháy", "chữa", "cháy"),
    "ubnd": ("ủy", "ban", "nhân", "dân"),
    "hđnd": ("hội", "đồng", "nhân", "dân"),
    "gplx": ("giấy", "phép", "lái", "xe"),
    "atgt": ("an", "toàn", "giao", "thông"),
    "qsdđ": ("quyền", "sử", "dụng", "đất"),
    "bđs": ("bất", "động", "sản"),
}

VIEW_NAMES = ("packed", "article", "metadata", "recovery", "short", "unknown")
VIEW_TO_CODE = {name: index for index, name in enumerate(VIEW_NAMES)}
LEXICAL_PROFILE = {
    "name": "vi_legal_fields_v1",
    "unicode_normalization": "NFC+casefold",
    "stemming": False,
    "lemmatization": False,
    "namespaces": [
        "base",
        "body",
        "hierarchy",
        "title",
        "accent_folded",
        "adjacent_bigram",
        "document_number",
        "legal_reference",
        "optional_vietnamese_word_segmentation",
        "common_legal_abbreviation_expansion",
    ],
}


def normalize_inline(value: Any) -> str:
    text = unicodedata.normalize("NFC", str(value or ""))
    text = ZERO_WIDTH_RE.sub("", text).replace("\xa0", " ")
    return MULTISPACE_RE.sub(" ", text).strip()


def normalize_query(value: Any) -> str:
    lines = [normalize_inline(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def fold_vietnamese(value: Any) -> str:
    value = normalize_inline(value).casefold().replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", value)
        if unicodedata.category(character) != "Mn"
    )


def normalize_code(value: Any) -> str:
    text = fold_vietnamese(normalize_inline(value).translate(CODE_PUNCT_TRANSLATION))
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"-+", "-", text)
    text = re.sub(r"/+", "/", text)
    return text.strip(".,;:")


def numeric_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def infer_view(chunk_id: Any) -> str:
    value = str(chunk_id or "")
    if value.endswith("__metadata"):
        return "metadata"
    if re.search(r"__p\d+$", value):
        return "packed"
    if re.search(r"__l\d+$", value):
        return "article"
    if re.search(r"__r\d+$", value):
        return "recovery"
    if re.search(r"__s\d+$", value):
        return "short"
    return "unknown"


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {source}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL row must be an object at {source}:{line_number}"
                )
            yield value


def iter_retrieval_rows(path: str | Path) -> Iterator[dict[str, str]]:
    seen_chunk_ids: set[str] = set()
    for line_number, row in enumerate(iter_jsonl(path), 1):
        if set(row) != RETRIEVAL_FIELDS:
            raise ValueError(
                f"Retrieval schema mismatch at row {line_number}: "
                f"expected {sorted(RETRIEVAL_FIELDS)}, got {sorted(row)}"
            )
        chunk_id = normalize_inline(row["chunk_id"])
        context_id = normalize_inline(row["context_id"])
        text = normalize_query(row["text"])
        if not chunk_id or not context_id or not text:
            raise ValueError(f"Empty retrieval field at row {line_number}")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"Duplicate chunk_id at row {line_number}: {chunk_id}")
        seen_chunk_ids.add(chunk_id)
        if not chunk_id.startswith(f"ctx{context_id}__"):
            raise ValueError(
                f"chunk_id/context_id mismatch at row {line_number}: "
                f"{chunk_id!r} vs {context_id!r}"
            )
        yield {"chunk_id": chunk_id, "context_id": context_id, "text": text}


def parse_tagged_legal_text(text: Any) -> dict[str, str]:
    title_parts: list[str] = []
    number_parts: list[str] = []
    hierarchy_parts: list[str] = []
    body_parts: list[str] = []
    metadata_parts: list[str] = []
    active_tag = ""
    for raw_line in normalize_query(text).split("\n"):
        match = TAG_RE.match(raw_line)
        if match:
            active_tag = normalize_inline(match.group(1)).upper()
            value = normalize_inline(match.group(2))
        else:
            value = normalize_inline(raw_line)
        if not value:
            continue
        if active_tag in {
            "VĂN BẢN",
            "TIÊU ĐỀ CHÍNH THỨC",
            "TIÊU ĐỀ SINH",
            "TIÊU ĐỀ GỢI Ý",
        }:
            title_parts.append(value)
        elif active_tag == "SỐ/KÝ HIỆU":
            number_parts.append(value)
        elif active_tag in {"PHÂN CẤP", "CHỦ ĐỀ/ĐIỀU MỤC", "TỪ KHÓA"}:
            hierarchy_parts.append(value)
        elif active_tag == "NỘI DUNG":
            body_parts.append(value)
        else:
            metadata_parts.append(value)

    def unique_join(values: Sequence[str]) -> str:
        return "\n".join(dict.fromkeys(item for item in values if item))

    metadata = unique_join(metadata_parts)
    body = unique_join(body_parts) or metadata
    return {
        "title": unique_join(title_parts),
        "document_number": unique_join(number_parts),
        "document_number_normalized": normalize_code(number_parts[0])
        if number_parts
        else "",
        "hierarchy": unique_join(hierarchy_parts),
        "body": body,
        "metadata": metadata,
    }


class SimpleWordSegmenter:
    name = "simple"

    def segment(self, text: str) -> str:
        return normalize_inline(text).casefold()


class UndertheseaWordSegmenter:
    name = "underthesea"

    def __init__(self) -> None:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "underthesea is not installed; install requirements-vietnamese.txt "
                "or use --word-segmenter simple"
            ) from exc
        self._word_tokenize = word_tokenize

    def segment(self, text: str) -> str:
        return self._word_tokenize(normalize_inline(text).casefold(), format="text")


class VnCoreNLPWordSegmenter:
    name = "vncorenlp"

    def __init__(self, model_dir: str | Path | None) -> None:
        if model_dir is None:
            raise ValueError("--vncorenlp-dir is required for WORD_SEGMENTER=vncorenlp")
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise RuntimeError(
                "py_vncorenlp is not installed; install requirements-vietnamese.txt "
                "or use --word-segmenter simple"
            ) from exc
        self._model = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(model_dir),
        )

    def segment(self, text: str) -> str:
        return " ".join(self._model.word_segment(normalize_inline(text).casefold()))


def build_word_segmenter(name: str, vncorenlp_dir: str | Path | None = None):
    if name == "underthesea":
        return UndertheseaWordSegmenter()
    if name == "vncorenlp":
        return VnCoreNLPWordSegmenter(vncorenlp_dir)
    if name != "simple":
        raise ValueError(f"Unknown word segmenter: {name}")
    return SimpleWordSegmenter()


def word_tokens(value: Any, segmenter: Any | None = None) -> list[str]:
    text = unicodedata.normalize("NFC", str(value or "")).casefold()
    if segmenter is not None:
        text = segmenter.segment(text)
    output: list[str] = []
    for token in WORD_RE.findall(text):
        output.append(token)


        if "_" in token:
            output.extend(part for part in token.split("_") if part)
    return output


def expand_legal_abbreviations(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        output.append(value)
        output.extend(LEGAL_ABBREVIATIONS.get(value, ()))
    return output


def _field_tokens(prefix: str, values: Sequence[str]) -> list[str]:
    return [
        f"{prefix}¦{value}"
        for value in values
        if value not in VIETNAMESE_QUERY_STOPWORDS
    ]


def _bigram_tokens(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for left, right in zip(values, values[1:]):
        if left in VIETNAMESE_QUERY_STOPWORDS and right in VIETNAMESE_QUERY_STOPWORDS:
            continue
        output.append(f"g¦{left}~{right}")
    return output


def _reference_tokens(value: Any) -> list[str]:
    output: list[str] = []
    for kind, number in LEGAL_REFERENCE_RE.findall(normalize_query(value)):
        normalized_kind = fold_vietnamese(kind).replace(" ", "_")
        normalized_number = fold_vietnamese(number)
        output.append(f"r¦{normalized_kind}_{normalized_number}")
    return list(dict.fromkeys(output))


def lexical_document_tokens(
    text: Any,
    profile: str = "enhanced",
    segmenter: Any | None = None,
) -> list[str]:
    if profile not in {"plain", "enhanced"}:
        raise ValueError(f"Unknown lexical profile: {profile}")
    fields = parse_tagged_legal_text(text)
    base_text = "\n".join(
        item
        for item in (
            fields["title"],
            fields["document_number"],
            fields["hierarchy"],
            fields["body"],
            fields["metadata"],
        )
        if item
    )
    base = expand_legal_abbreviations(word_tokens(base_text, segmenter))
    tokens = [f"u¦{value}" for value in base]
    if profile == "plain":
        return tokens

    title = expand_legal_abbreviations(word_tokens(fields["title"], segmenter))
    hierarchy = expand_legal_abbreviations(word_tokens(fields["hierarchy"], segmenter))
    body = expand_legal_abbreviations(word_tokens(fields["body"], segmenter))
    tokens.extend(_field_tokens("t", title))
    tokens.extend(_field_tokens("h", hierarchy))
    tokens.extend(_field_tokens("b", body))
    tokens.extend(_bigram_tokens(title))
    tokens.extend(_bigram_tokens(hierarchy))
    tokens.extend(_bigram_tokens(body))
    for value in base:
        folded = fold_vietnamese(value)
        if folded and folded != value:
            tokens.append(f"a¦{folded}")
    document_number = fields["document_number_normalized"]
    if document_number:
        tokens.append(f"c¦{document_number}")
    tokens.extend(_reference_tokens(text))
    return tokens


def lexical_query_tokens(
    question: Any,
    profile: str = "enhanced",
    segmenter: Any | None = None,
) -> list[str]:
    if profile not in {"plain", "enhanced"}:
        raise ValueError(f"Unknown lexical profile: {profile}")
    words = expand_legal_abbreviations(word_tokens(question, segmenter))
    tokens = [f"u¦{value}" for value in words]
    if profile == "plain":
        return tokens

    focused = [value for value in words if value not in VIETNAMESE_QUERY_STOPWORDS]


    for prefix in ("b", "h", "t"):
        tokens.extend(f"{prefix}¦{value}" for value in focused)
    tokens.extend(_bigram_tokens(words))
    for value in words:
        folded = fold_vietnamese(value)
        if folded and folded != value:
            tokens.append(f"a¦{folded}")
    for match in DOCUMENT_NUMBER_RE.findall(normalize_query(question)):
        normalized = normalize_code(match)
        if normalized:
            tokens.append(f"c¦{normalized}")
    tokens.extend(_reference_tokens(question))
    return tokens


def lexical_tokens_to_text(tokens: Sequence[str]) -> str:
    return " ".join(tokens)


def lexical_profile_signature(profile: str, word_segmenter: str = "simple") -> str:
    value = dict(LEXICAL_PROFILE)
    value["selected_profile"] = profile
    value["word_segmenter"] = word_segmenter
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_queries(path: str | Path) -> list[dict[str, str]]:
    queries: list[dict[str, str]] = []
    seen: set[str] = set()
    for row_number, row in enumerate(iter_jsonl(path), 1):
        if set(row) != {"query_id", "question"}:
            raise ValueError(
                f"Query schema mismatch at row {row_number}: got {sorted(row)}"
            )
        query_id = normalize_inline(row["query_id"])
        question = normalize_query(row["question"])
        if not query_id or not question:
            raise ValueError(f"Empty query at row {row_number}")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id: {query_id}")
        seen.add(query_id)
        queries.append({"query_id": query_id, "question": question})
    return queries


def select_answer_ids(
    hits: Sequence[Mapping[str, Any]],
    top_k: int = 5,
    score_ratio_cutoff: float = 0.0,
) -> list[str]:
    if not 1 <= top_k <= 5:
        raise ValueError("The official evaluator requires 1 <= top_k <= 5")
    if not 0.0 <= score_ratio_cutoff <= 1.0:
        raise ValueError("score_ratio_cutoff must be between 0 and 1")
    if not hits:
        return []
    best_score = float(hits[0]["score"])
    selected: list[str] = []
    seen: set[str] = set()
    for hit in hits:
        context_id = str(hit["context_id"])
        score = float(hit["score"])
        if context_id in seen:
            continue
        if (
            selected
            and score_ratio_cutoff
            and best_score > 0.0
            and score < best_score * score_ratio_cutoff
        ):
            break
        seen.add(context_id)
        selected.append(context_id)
        if len(selected) == top_k:
            break
    return selected


def build_submission(
    query_order: Sequence[str],
    answers: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, list[str]]]:
    submission: dict[str, dict[str, list[str]]] = {}
    for query_id in query_order:
        values = list(dict.fromkeys(str(value) for value in answers.get(query_id, [])))
        submission[str(query_id)] = {"answer": values[:5]}
    validate_submission(submission, expected_query_ids=query_order)
    return submission


def validate_submission(
    submission: Mapping[str, Any],
    expected_query_ids: Sequence[str] | None = None,
) -> None:
    if expected_query_ids is not None and set(submission) != set(expected_query_ids):
        raise ValueError("Submission query IDs do not exactly match the query file")
    for query_id, value in submission.items():
        if not isinstance(value, Mapping) or set(value) != {"answer"}:
            raise ValueError(f"Invalid submission value for query {query_id}")
        answers = value["answer"]
        if not isinstance(answers, list) or not 1 <= len(answers) <= 5:
            raise ValueError(f"Query {query_id} must contain between 1 and 5 answers")
        normalized = [str(item) for item in answers]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Query {query_id} contains duplicate context IDs")


def atomic_write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def atomic_write_lines(path: str | Path, values: Iterable[str]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for value in values:
                text = str(value)
                if "\n" in text or "\r" in text:
                    raise ValueError("Sidecar identifiers must not contain newlines")
                handle.write(text + "\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
    return count


def atomic_save_npy(path: str | Path, values: Any) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def expected_retrieval_count(corpus_path: str | Path) -> int | None:
    corpus = Path(corpus_path).resolve()
    for manifest_path in (
        corpus.parent.parent / "manifest.json",
        corpus.parent / "manifest.json",
    ):
        if not manifest_path.is_file():
            continue
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        value = manifest.get("counts", {}).get("retrieval_records")
        if value is not None:
            return int(value)
    return None


def l2_normalize(vectors: Any):
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for BGE-M3 embeddings") from exc
    values = np.asarray(vectors, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D embedding matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(norms)):
        raise ValueError("Embedding matrix contains NaN or infinity")
    if np.any(norms <= 0):
        raise ValueError("Embedding matrix contains a zero vector")
    normalized = np.ascontiguousarray(values / norms, dtype=np.float32)
    if not np.allclose(np.linalg.norm(normalized, axis=1), 1.0, atol=1e-4):
        raise ValueError("Failed to normalize embeddings")
    return normalized


def finite_score(value: Any) -> float:
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"Non-finite retrieval score: {score}")
    return score


class ChunkTableBuilder:

    def __init__(self) -> None:
        self.contexts: list[str] = []
        self._context_to_ordinal: dict[str, int] = {}
        self.context_ordinals = array("I")
        self.view_codes = array("B")
        self.chunk_ids: list[str] = []

    def add(self, chunk_id: str, context_id: str) -> None:
        ordinal = self._context_to_ordinal.get(context_id)
        if ordinal is None:
            ordinal = len(self.contexts)
            self._context_to_ordinal[context_id] = ordinal
            self.contexts.append(context_id)
        view = infer_view(chunk_id)
        self.context_ordinals.append(ordinal)
        self.view_codes.append(VIEW_TO_CODE[view])
        self.chunk_ids.append(chunk_id)

    def __len__(self) -> int:
        return len(self.chunk_ids)

    def save(self, directory: str | Path) -> None:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("numpy is required") from exc
        target = Path(directory)
        atomic_write_json(target / "contexts.json", self.contexts)
        atomic_save_npy(
            target / "context_ordinals.npy",
            np.asarray(self.context_ordinals, dtype=np.int32),
        )
        atomic_save_npy(
            target / "view_codes.npy", np.asarray(self.view_codes, dtype=np.uint8)
        )
        written = atomic_write_lines(target / "chunk_ids.txt", self.chunk_ids)
        if written != len(self):
            raise AssertionError("Chunk sidecar write count mismatch")


@dataclass(frozen=True)
class ChunkTable:
    contexts: list[str]
    context_ordinals: Any
    view_codes: Any
    chunk_ids: list[str]

    def __len__(self) -> int:
        return len(self.chunk_ids)


def load_chunk_table(directory: str | Path, mmap: bool = True) -> ChunkTable:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required") from exc
    source = Path(directory)
    with (source / "contexts.json").open("r", encoding="utf-8") as handle:
        contexts = json.load(handle)
    if not isinstance(contexts, list) or not all(
        isinstance(item, str) for item in contexts
    ):
        raise ValueError("Invalid contexts.json sidecar")
    mmap_mode = "r" if mmap else None
    ordinals = np.load(
        source / "context_ordinals.npy", mmap_mode=mmap_mode, allow_pickle=False
    )
    view_codes = np.load(
        source / "view_codes.npy", mmap_mode=mmap_mode, allow_pickle=False
    )
    with (source / "chunk_ids.txt").open("r", encoding="utf-8") as handle:
        chunk_ids = [line.rstrip("\n\r") for line in handle]
    if not (len(ordinals) == len(view_codes) == len(chunk_ids)):
        raise ValueError("Local index sidecars have different row counts")
    if len(ordinals) and (
        int(ordinals.min()) < 0 or int(ordinals.max()) >= len(contexts)
    ):
        raise ValueError("context_ordinals.npy contains an invalid context ordinal")
    if len(view_codes) and int(view_codes.max()) >= len(VIEW_NAMES):
        raise ValueError("view_codes.npy contains an invalid view code")
    return ChunkTable(contexts, ordinals, view_codes, chunk_ids)


def validate_view_weights(weights: Mapping[str, float]) -> dict[str, float]:
    output = {name: float(weights.get(name, 1.0)) for name in VIEW_NAMES}
    if any(not math.isfinite(value) or value <= 0.0 for value in output.values()):
        raise ValueError("All view weights must be finite and positive")
    return output


def aggregate_chunk_hits(
    indices: Sequence[Any],
    scores: Sequence[Any],
    table: ChunkTable,
    context_pool: int,
    view_weights: Mapping[str, float] | None = None,
    skip_nonpositive: bool = False,
) -> list[dict[str, Any]]:
    if context_pool <= 0:
        raise ValueError("context_pool must be positive")
    weights = validate_view_weights(view_weights or {})
    best: dict[int, dict[str, Any]] = {}
    for row_value, score_value in zip(indices, scores):
        row_id = int(row_value)
        if row_id < 0:
            continue
        if row_id >= len(table):
            raise ValueError(f"Search returned out-of-range chunk row {row_id}")
        raw_score = finite_score(score_value)
        if skip_nonpositive and raw_score <= 0.0:
            continue
        view = VIEW_NAMES[int(table.view_codes[row_id])]
        weight = weights[view]
        adjusted = raw_score * weight if raw_score >= 0.0 else raw_score / weight
        context_ordinal = int(table.context_ordinals[row_id])
        candidate = {
            "context_id": table.contexts[context_ordinal],
            "chunk_id": table.chunk_ids[row_id],
            "view": view,
            "row_id": row_id,
            "raw_score": raw_score,
            "view_weight": weight,
            "score": adjusted,
        }
        previous = best.get(context_ordinal)
        if previous is None or (
            adjusted,
            raw_score,
            -row_id,
        ) > (
            float(previous["score"]),
            float(previous["raw_score"]),
            -int(previous["row_id"]),
        ):
            best[context_ordinal] = candidate
    ranked = sorted(
        best.values(),
        key=lambda item: (
            -float(item["score"]),
            numeric_sort_key(item["context_id"]),
            str(item["chunk_id"]),
        ),
    )
    return ranked[:context_pool]


def fill_with_fallback(
    hits: list[dict[str, Any]],
    contexts: Sequence[str],
    minimum_count: int,
) -> list[dict[str, Any]]:
    seen = {str(item["context_id"]) for item in hits}
    for context_id in sorted((str(item) for item in contexts), key=numeric_sort_key):
        if len(hits) >= minimum_count:
            break
        if context_id in seen:
            continue
        hits.append(
            {
                "context_id": context_id,
                "chunk_id": "",
                "view": "fallback",
                "row_id": -1,
                "raw_score": 0.0,
                "view_weight": 1.0,
                "score": 0.0,
            }
        )
        seen.add(context_id)
    return hits


def create_build_directory(target: str | Path, recreate: bool) -> Path:
    destination = Path(target).resolve()
    if destination.exists() and not recreate:
        raise FileExistsError(
            f"Index directory already exists: {destination}. Pass --recreate to replace it."
        )
    if destination == Path(destination.anchor) or len(destination.parts) < 3:
        raise ValueError(f"Refusing unsafe index directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.building.", dir=destination.parent
        )
    )


def commit_build_directory(
    build_dir: str | Path, target: str | Path, recreate: bool
) -> None:
    source = Path(build_dir).resolve()
    destination = Path(target).resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists():
        if not recreate:
            raise FileExistsError(destination)
        if destination == Path(destination.anchor) or len(destination.parts) < 3:
            raise ValueError(f"Refusing unsafe replacement target: {destination}")
        backup = destination.with_name(f".{destination.name}.old.{uuid.uuid4().hex}")
        os.replace(destination, backup)
        try:
            os.replace(source, destination)
        except Exception:
            os.replace(backup, destination)
            raise
        shutil.rmtree(backup)
        return
    os.replace(source, destination)


def discard_build_directory(path: str | Path) -> None:
    source = Path(path)
    if source.is_dir() and source.name.startswith(".") and ".building." in source.name:
        shutil.rmtree(source)


def load_index_manifest(
    directory: str | Path, backend: str | None = None
) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if (
        not isinstance(value, dict)
        or value.get("format_version") != INDEX_FORMAT_VERSION
    ):
        raise ValueError(f"Unsupported or invalid local index manifest: {path}")
    if value.get("status") != "complete":
        raise ValueError(f"Local index is not complete: {path}")
    if backend is not None and value.get("backend") != backend:
        raise ValueError(
            f"Index backend mismatch: expected {backend!r}, got {value.get('backend')!r}"
        )
    return value


def index_file_inventory(directory: str | Path) -> dict[str, int]:
    root = Path(directory)
    return {
        str(path.relative_to(root)): path.stat().st_size
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def canonical_json_value(value: Any) -> Any:
    if is_dataclass(value):
        return canonical_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): canonical_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [canonical_json_value(item) for item in value]
    if isinstance(value, set):
        return sorted(canonical_json_value(item) for item in value)
    return value


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {source}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON mapping: {source}")
    return value


def artifact_provenance(path: str | Path) -> dict[str, str]:
    source = Path(path)
    return {"path": str(source.resolve()), "sha256": sha256_file(source)}


def snapshot_file_inventory(directory: str | Path) -> dict[str, str]:
    root = Path(directory).resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    inventory: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        inventory[path.relative_to(root).as_posix()] = sha256_file(path)
    if not inventory:
        raise ValueError(f"Directory has no files: {root}")
    return inventory


def verify_file_inventory(
    directory: str | Path, expected_inventory: Mapping[str, str]
) -> dict[str, str]:
    actual = snapshot_file_inventory(directory)
    expected = dict(expected_inventory)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(
            key
            for key in set(actual) & set(expected)
            if actual[key] != expected[key]
        )
        raise ValueError(
            "Model runtime closure differs from pinned inventory: "
            f"missing={missing} extra={extra} changed={changed}"
        )
    return actual


def count_safetensors_parameters(path: str | Path) -> int:
    target = Path(path)
    with target.open("rb") as handle:
        encoded_size = handle.read(8)
        if len(encoded_size) != 8:
            raise ValueError(f"Invalid safetensors header: {target}")
        header_size = int.from_bytes(encoded_size, "little")
        if header_size < 2 or header_size > target.stat().st_size - 8:
            raise ValueError(f"Invalid safetensors header size: {target}")
        try:
            header = json.loads(handle.read(header_size))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid safetensors JSON header: {target}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors mapping: {target}")
    parameter_count = 0
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid safetensors tensor metadata: {name}")
        shape = metadata.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"Safetensors tensor lacks shape: {name}")
        elements = 1
        for dimension in shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 1
            ):
                raise ValueError(f"Safetensors shape must be positive: {name}")
            elements *= dimension
        parameter_count += elements
    if parameter_count < 1:
        raise ValueError(f"Safetensors checkpoint has no parameters: {target}")
    return parameter_count


def checkpoint_parameter_count(
    model_dir: str | Path,
) -> tuple[int, dict[str, str]]:
    resolved = Path(model_dir).resolve()
    single = resolved / "model.safetensors"
    if single.is_file():
        return count_safetensors_parameters(single), {
            "model.safetensors": sha256_file(single)
        }
    index_path = resolved / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Expected model.safetensors or model.safetensors.index.json in {resolved}"
        )
    index = read_json_mapping(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError("Safetensors index has no weight_map")
    shard_names = sorted({str(value) for value in weight_map.values()})
    if any(
        Path(name).is_absolute()
        or ".." in Path(name).parts
        or not name.endswith(".safetensors")
        for name in shard_names
    ):
        raise ValueError("Safetensors index has unsafe shard path")
    files = {"model.safetensors.index.json": sha256_file(index_path)}
    total = 0
    for name in shard_names:
        shard = resolved / name
        if not shard.is_file():
            raise FileNotFoundError(shard)
        files[name] = sha256_file(shard)
        total += count_safetensors_parameters(shard)
    return total, files


def load_ranked_jsonl(
    path: str | Path,
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    source = Path(path)
    order: list[str] = []
    output: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(iter_jsonl(source), 1):
        query_id = str(row.get("query_id", ""))
        ranked = row.get("ranked")
        if not query_id or not isinstance(ranked, list):
            raise ValueError(f"Invalid ranking at {source}:{row_number}")
        if query_id in output:
            raise ValueError(f"Duplicate query_id in {source}: {query_id}")
        evidence: dict[str, dict[str, Any]] = {}
        hits = row.get("hits", [])
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, Mapping):
                    continue
                context_id = str(hit.get("context_id", ""))
                if context_id and context_id not in evidence:
                    evidence[context_id] = dict(hit)
        output[query_id] = {
            "question": str(row.get("question", "")),
            "ranked": list(
                dict.fromkeys(str(item) for item in ranked if str(item))
            ), 
            "evidence": evidence,
        }
        order.append(query_id)
    if not order:
        raise ValueError(f"Empty ranked file: {source}")
    return order, output
