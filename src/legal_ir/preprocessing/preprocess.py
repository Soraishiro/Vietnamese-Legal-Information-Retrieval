"""Dựng corpus truy hồi và file câu hỏi từ văn bản pháp luật gốc.

File này làm gì
----------------
Đọc các file văn bản pháp luật, tách thành đơn vị pháp lý (chương/điều/khoản/điểm/giáp),
ghép lại thành chunk phục vụ truy hồi, rồi ghi ra một workspace gồm corpus, câu hỏi và
các báo cáo chất lượng. Có hai lệnh: `build` dựng workspace, `validate` kiểm tra
workspace đã có.

Pipeline stage
--------------
Bước đầu tiên, trước mọi bước index và truy hồi.

Input
-----
  build: --contexts-dir (thư mục văn bản gốc), --queries (file câu hỏi),
         --labeled-queries (chỉ các file ở đây mới được đọc trường "answer"),
         --output, cùng các tham số kích thước chunk và profile truy hồi
  validate: --workspace (thư mục workspace cần kiểm)

Output
-------
  <output>/corpus/retrieval_chunks.jsonl   chunk phục vụ truy hồi
  <output>/corpus/corpus_raw.jsonl        văn bản gốc đã chuẩn hoá
  <output>/queries/<tập>.jsonl            câu hỏi, mỗi dòng query_id + question
  <output>/manifest.json                  tham số đã dùng, số lượng, thống kê chất lượng

Model/class
-----------
Có thể dùng underthesea hoặc vncorenlp để tách từ nếu cài, nếu không thì tự tách từ bằng
quy tắc đơn giản.

Command
-------
  python preprocess.py build --contexts-dir contexts/corpus \
      --queries train.json queries.json --labeled-queries train.json \
      --output <W>
  python preprocess.py validate --workspace <W>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import dataclasses
import datetime as dt
import hashlib
import html
import json
import logging
import math
import os
import re
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence
from urllib.parse import unquote, urlparse


LOGGER = logging.getLogger("legalir-preprocess")
PIPELINE_VERSION = "5.3.0-lossless-recovery-quarantine"


ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
MULTISPACE_RE = re.compile(r"[^\S\n]+")
HTML_TAG_RE = re.compile(r"<[A-Za-z!/][^>]*>")
ELLIPSIS_ONLY_RE = re.compile(r"^(?:\.{3,}|…+)(?:\s+(?:\.{3,}|…+))*$")

CHAPTER_RE = re.compile(
    r"^(?P<kind>PHẦN|Phần|CHƯƠNG|Chương)\s+"
    r"(?P<label>[IVXLCDM\d]+[A-Za-zĐđ]?)"
    r"(?:\s*[\-–—:.)]\s*|\s+)?(?P<title>.*)$"
)
SECTION_RE = re.compile(
    r"^(?P<kind>MỤC|Mục|TIỂU MỤC|Tiểu mục)\s+"
    r"(?P<label>[IVXLCDM\d]+[A-Za-zĐđ]?)"
    r"(?:\s*[\-–—:.)]\s*|\s+)?(?P<title>.*)$"
)
ARTICLE_RE = re.compile(
    r"^(?P<kind>ĐIỀU|Điều)\s+"
    r"(?P<label>\d+[A-Za-zĐđ]?)"
    r"(?:\s*[\-–—:.)]\s*|\s+)?(?P<title>.*)$"
)
APPENDIX_RE = re.compile(
    r"^(?P<kind>PHỤ\s+LỤC|Phụ\s+lục)\s*"
    r"(?P<label>[IVXLCDM\dA-Za-zĐđ.-]*)"
    r"(?:\s*[\-–—:.)]\s*|\s+)?(?P<title>.*)$"
)
CLAUSE_RE = re.compile(r"^(?P<label>\d+[A-Za-zĐđ]?)\.\s+(?P<body>.+)$")
POINT_RE = re.compile(
    r"^(?P<label>[A-Za-zĐđ])\s*[\).]\s+(?P<body>.+)$"
)
ROMAN_HEADING_RE = re.compile(
    r"^(?P<label>[IVXLCDM]{1,8})\.\s+(?P<body>.+)$"
)
NUMERIC_HEADING_RE = re.compile(
    r"^(?P<label>\d+(?:\.\d+){1,5})\.?\s+(?P<body>.+)$"
)
BULLET_RE = re.compile(r"^(?P<label>[-–—•+])\s+(?P<body>.+)$")

DOCUMENT_NUMBER_RES = (
    re.compile(
        r"\bSố\s*:\s*(?P<number>"
        r"\d+(?:/\d{2,4})?/[A-ZÀ-ỸĐ0-9.-]+(?:-[A-ZÀ-ỸĐ0-9.-]+)*)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<number>TCVN\s+(?:ISO(?:/IEC)?\s+)?\d+(?::\d{4})?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?P<number>\d+/\d{4}/[A-ZÀ-ỸĐ0-9.-]+(?:-[A-ZÀ-ỸĐ0-9.-]+)*)\b",
        re.IGNORECASE,
    ),
)
DATE_RE = re.compile(
    r"\bngày\s+(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})"
    r"\s+năm\s+(?P<year>\d{4})\b",
    re.IGNORECASE,
)
LEGAL_REFERENCE_RE = re.compile(
    r"(?:(?:khoản)\s+(?P<clause>\d+[A-Za-zĐđ]?)\s+)?"
    r"(?:điểm\s+(?P<point>[A-Za-zĐđ])\s+)?"
    r"(?:Điều|điều)\s+(?P<article>\d+[A-Za-zĐđ]?)"
    r"(?:\s+(?:của\s+)?(?P<law_type>Luật|Nghị định|Thông tư|Quyết định))?"
    r"(?:\s+(?:số\s+)?(?P<law_number>"
    r"\d+(?:/\d{2,4})?/[A-ZÀ-ỸĐ0-9.-]+(?:-[A-ZÀ-ỸĐ0-9.-]+)*))?",
    re.UNICODE,
)

DOCUMENT_TYPE_LINES = {
    "BỘ LUẬT",
    "CHỈ THỊ",
    "CÔNG VĂN",
    "HIẾN PHÁP",
    "HIỆP ĐỊNH",
    "HƯỚNG DẪN",
    "KẾ HOẠCH",
    "LUẬT",
    "NGHỊ ĐỊNH",
    "NGHỊ QUYẾT",
    "PHÁP LỆNH",
    "QUY CHẾ",
    "QUYẾT ĐỊNH",
    "THÔNG BÁO",
    "THÔNG TƯ",
    "THÔNG TƯ LIÊN TỊCH",
    "TIÊU CHUẨN QUỐC GIA",
    "QUY CHUẨN KỸ THUẬT QUỐC GIA",
}
PREAMBLE_PREFIXES = (
    "Căn cứ ",
    "Theo đề nghị ",
    "Xét đề nghị ",
    "QUYẾT NGHỊ",
    "QUYẾT ĐỊNH:",
)
NON_CONTENT_HEADINGS = {
    "MỤC LỤC",
    "TABLE OF CONTENTS",
}
TRAILING_REFERENCE_HEADINGS = {
    "THƯ MỤC TÀI LIỆU THAM KHẢO",
    "TÀI LIỆU THAM KHẢO",
}


ADMIN_START_RE = re.compile(r"(?im)(?:^|\n)\s*Nơi\s+nhận\s*:")
ADMIN_RESTART_RE = re.compile(
    r"(?im)^\s*(?:"
    r"CHƯƠNG\s+TRÌNH(?:\s+HÀNH\s+ĐỘNG)?|NỘI\s+QUY|"
    r"QUY\s+CHẾ|PHỤ\s+LỤC|ĐIỀU\s+LỆ|ĐỀ\s+ÁN|DANH\s+MỤC|"
    r"THỂ\s+LỆ|QUY\s+ĐỊNH|QUY\s+TRÌNH|KẾ\s+HOẠCH|HƯỚNG\s+DẪN|"
    r"TIÊU\s+CHUẨN|QUY\s+CHUẨN|NỘI\s+DUNG|TIÊU\s+CHÍ|"
    r"MẪU(?:\s+SỐ)?\b|BẢNG(?:\s+\d+)?\b|BIỂU(?:\s+\d+)?\b|"
    r"(?:PHẦN|CHƯƠNG|MỤC)\s+[IVXLCDM\d]+\b|"
    r"Điều\s+\d+[A-Za-zĐđ]?\b"
    r")"
)


ADMIN_INLINE_RESTART_RE = re.compile(
    r"(?i)\b(?:"
    r"CHƯƠNG\s+TRÌNH(?:\s+HÀNH\s+ĐỘNG)?|NỘI\s+QUY|"
    r"QUY\s+CHẾ|PHỤ\s+LỤC(?:\s+[IVXLCDM\d]+)?|ĐIỀU\s+LỆ|"
    r"ĐỀ\s+ÁN|DANH\s+MỤC|THỂ\s+LỆ|QUY\s+ĐỊNH|"
    r"QUY\s+TRÌNH|KẾ\s+HOẠCH|HƯỚNG\s+DẪN|"
    r"TIÊU\s+CHUẨN|QUY\s+CHUẨN|NỘI\s+DUNG|TIÊU\s+CHÍ|"
    r"MẪU(?:\s+SỐ)?(?:\s+\d+)?|BẢNG(?:\s+\d+)?|BIỂU(?:\s+\d+)?"
    r")\b"
)
ADMIN_ATTACHMENT_CUE_RE = re.compile(
    r"(?i)\b(?:ban\s+hành\s+)?kèm\s+theo\b"
)
ADMIN_SIGNATURE_CUE_RE = re.compile(
    r"(?i)(?:"
    r"\bTM\s*\.|\bKT\s*\.|\bTL\s*\.|\bTUQ\s*\.|"
    r"\bQ\s*\.\s*(?:BỘ|CỤC)\s+TRƯỞNG\b|"
    r"\b(?:THỦ\s+TƯỚNG|BỘ\s+TRƯỞNG|THỨ\s+TRƯỞNG|"
    r"TỔNG\s+CỤC\s+TRƯỞNG|CỤC\s+TRƯỞNG|VỤ\s+TRƯỞNG|"
    r"CHỦ\s+TỊCH|PHÓ\s+CHỦ\s+TỊCH|GIÁM\s+ĐỐC|"
    r"PHÓ\s+GIÁM\s+ĐỐC)\b|\bĐÃ\s+KÝ\b"
    r")"
)


MAX_UNCERTAIN_ADMIN_REMOVAL_PPM = 350_000


@dataclass(slots=True)
class BuildConfig:
    contexts_dir: Path
    query_files: list[Path]
    output_dir: Path
    labeled_query_files: list[Path] = dataclasses.field(default_factory=list)
    short_body_chars: int = 450
    retrieval_body_chars: int = 1000
    retrieval_overlap_chars: int = 120
    long_body_chars: int = 2000
    long_overlap_units: int = 1
    long_overlap_chars: int = 200
    max_prefix_chars: int = 500
    max_workers: int = max(1, min(16, (os.cpu_count() or 4)))
    word_segmenter: str = "simple"
    vncorenlp_dir: Optional[Path] = None
    metadata_jsonl: Optional[Path] = None
    include_raw_passage: bool = False
    allow_partial_json: bool = False
    strict: bool = False
    pretty_manifest: bool = True
    analysis_level: str = "none"
    retrieval_profile: str = "quality"
    metadata_max_chars: int = 1800
    empty_labeled_policy: str = "quarantine"

    def validate(self) -> None:
        if self.short_body_chars < 100:
            raise ValueError("short_body_chars must be >= 100")
        if self.retrieval_body_chars < self.short_body_chars:
            raise ValueError(
                "retrieval_body_chars must be >= short_body_chars"
            )
        if not 0 <= self.retrieval_overlap_chars < self.retrieval_body_chars:
            raise ValueError(
                "retrieval_overlap_chars must be >= 0 and smaller than "
                "retrieval_body_chars"
            )
        if self.long_body_chars < self.short_body_chars:
            raise ValueError(
                "long_body_chars must be >= short_body_chars"
            )
        if self.long_overlap_units < 0:
            raise ValueError("long_overlap_units must be >= 0")
        if self.long_overlap_chars < 0:
            raise ValueError("long_overlap_chars must be >= 0")
        if self.max_prefix_chars < 80:
            raise ValueError("max_prefix_chars must be >= 80")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        if self.analysis_level not in {"none", "full"}:
            raise ValueError("analysis_level must be one of: none, full")
        if self.retrieval_profile not in {"compact", "quality"}:
            raise ValueError(
                "retrieval_profile must be one of: compact, quality"
            )
        if self.metadata_max_chars < 300:
            raise ValueError("metadata_max_chars must be >= 300")
        if self.empty_labeled_policy not in {"quarantine", "error"}:
            raise ValueError(
                "empty_labeled_policy must be one of: quarantine, error"
            )


@dataclass(slots=True)
class SourceDocument:
    context_id: str
    source_file: str
    link: str
    name: str
    raw_passage: str
    normalized_passage: str
    paragraphs: list[str]
    title: str
    title_source: str
    document_type: str
    document_number: str
    promulgation_date: str
    effective_date: str
    issuing_authority: str
    source_domain: str
    content_sha256: str
    omission_markers: int
    administrative_recovery_paragraphs: list[str] = field(
        default_factory=list
    )
    exact_duplicate_of: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LegalUnit:
    unit_id: str
    context_id: str
    order: int
    kind: str
    label: str
    title: str
    text: str
    hierarchy: list[str]
    article_label: str = ""
    clause_label: str = ""
    point_label: str = ""
    indexable: bool = True
    excluded_reason: str = ""


@dataclass(slots=True)
class QueryRecord:
    dataset: str
    query_id: str
    question: str
    normalized_question: str
    answer_context_ids: list[str]
    source_file: str


@dataclass(slots=True)
class ParsedContext:
    document: Optional[SourceDocument]
    units: list[LegalUnit]
    anomalies: list[dict[str, Any]]


class WordSegmenter(Protocol):
    name: str

    def segment(self, text: str) -> str:
        ...


class SimpleWordSegmenter:
    name = "simple"

    def segment(self, text: str) -> str:
        return normalize_inline(text).lower()


class UndertheseaWordSegmenter:
    name = "underthesea"

    def __init__(self) -> None:
        try:
            from underthesea import word_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "underthesea is not installed. Install optional requirements "
                "or use --word-segmenter simple."
            ) from exc
        self._word_tokenize = word_tokenize

    def segment(self, text: str) -> str:
        return self._word_tokenize(
            normalize_inline(text).lower(),
            format="text",
        )


class VnCoreNLPWordSegmenter:
    name = "vncorenlp"

    def __init__(self, model_dir: Optional[Path]) -> None:
        if model_dir is None:
            raise ValueError(
                "--vncorenlp-dir is required when using VnCoreNLP"
            )
        try:
            import py_vncorenlp
        except ImportError as exc:
            raise RuntimeError(
                "py_vncorenlp is not installed. Install optional "
                "requirements or use --word-segmenter simple."
            ) from exc
        self._model = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=str(model_dir),
        )

    def segment(self, text: str) -> str:
        sentences = self._model.word_segment(normalize_inline(text).lower())
        return " ".join(sentences)


class _TextHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data:
            self.parts.append(data)

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, Optional[str]]],
    ) -> None:
        if tag.lower() in {"br", "p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"p", "div", "li", "tr", "h1", "h2", "h3"}:
            self.parts.append("\n")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def normalize_id(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_inline(text: str) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    value = ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\u00a0", " ")
    value = MULTISPACE_RE.sub(" ", value)
    return value.strip()


def strip_html(value: str) -> str:
    if not HTML_TAG_RE.search(value):
        return html.unescape(value)
    parser = _TextHTMLParser()
    parser.feed(value)
    parser.close()
    return html.unescape("".join(parser.parts))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def write_json(path: Path, value: Any, pretty: bool = True) -> None:
    if pretty:
        payload = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=False,
        )
    else:
        payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    atomic_write_text(path, payload + "\n")


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise
    return count


class AtomicJSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, self.temp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        self.handle = os.fdopen(
            fd,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        self.count = 0
        self._committed = False

    def write(self, record: Mapping[str, Any]) -> None:
        self.handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        self.handle.write("\n")
        self.count += 1

    def commit(self) -> None:
        if self._committed:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temp_name, self.path)
        self._committed = True

    def abort(self) -> None:
        if not self.handle.closed:
            self.handle.close()
        try:
            os.unlink(self.temp_name)
        except OSError:
            pass

    def __enter__(self) -> "AtomicJSONLWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.abort()


class LengthDistribution:

    def __init__(self) -> None:
        self.frequencies: Counter[int] = Counter()
        self.count = 0
        self.total = 0
        self.minimum: Optional[int] = None
        self.maximum = 0

    def add(self, value: int) -> None:
        value = int(value)
        self.frequencies[value] += 1
        self.count += 1
        self.total += value
        self.minimum = (
            value if self.minimum is None else min(self.minimum, value)
        )
        self.maximum = max(self.maximum, value)

    def percentile(self, fraction: float) -> int:
        if not self.count:
            return 0
        target = max(1, math.ceil(self.count * fraction))
        cumulative = 0
        for value in sorted(self.frequencies):
            cumulative += self.frequencies[value]
            if cumulative >= target:
                return value
        return self.maximum

    def summary(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.minimum or 0,
            "p50": self.percentile(0.50),
            "p90": self.percentile(0.90),
            "p99": self.percentile(0.99),
            "max": self.maximum,
            "mean": (
                round(self.total / self.count, 2) if self.count else 0.0
            ),
        }


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def recover_top_level_object_members(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    index = 0
    length = len(text)

    while index < length and text[index].isspace():
        index += 1
    if index >= length or text[index] != "{":
        raise ValueError("Partial JSON recovery supports top-level objects only")
    index += 1

    recovered: dict[str, Any] = {}
    while index < length:
        while index < length and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= length or text[index] == "}":
            break
        try:
            key, key_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        if not isinstance(key, str):
            break
        index = key_end
        while index < length and text[index].isspace():
            index += 1
        if index >= length or text[index] != ":":
            break
        index += 1
        while index < length and text[index].isspace():
            index += 1
        try:
            value, value_end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            break
        recovered[key] = value
        index = value_end
    return recovered


def load_json_resilient(path: Path, allow_partial: bool) -> tuple[Any, bool]:
    try:
        return load_json(path), False
    except json.JSONDecodeError:
        if not allow_partial:
            raise
        text = path.read_text(encoding="utf-8-sig")
        recovered = recover_top_level_object_members(text)
        if not recovered:
            raise
        LOGGER.warning(
            "Recovered %d complete records from truncated JSON: %s",
            len(recovered),
            path,
        )
        return recovered, True


def numeric_sort_key(value: str) -> tuple[int, Any]:
    normalized = normalize_id(value)
    if normalized.isdigit():
        return (0, int(normalized))
    return (1, normalized)


def is_all_caps_heading(text: str) -> bool:
    letters = [character for character in text if character.isalpha()]
    if len(letters) < 3:
        return False
    return all(not character.islower() for character in letters)


def classify_structural_line(text: str) -> str:
    value = normalize_inline(text)
    upper = value.upper().rstrip(":")
    if not value:
        return "empty"
    if ELLIPSIS_ONLY_RE.fullmatch(value):
        return "ellipsis"
    if upper in DOCUMENT_TYPE_LINES:
        return "document_type"
    if upper in NON_CONTENT_HEADINGS:
        return "non_content"
    if upper in TRAILING_REFERENCE_HEADINGS:
        return "references"
    if CHAPTER_RE.match(value):
        return "chapter"
    if SECTION_RE.match(value):
        return "section"
    if ARTICLE_RE.match(value):
        return "article"
    if APPENDIX_RE.match(value):
        return "appendix"
    if NUMERIC_HEADING_RE.match(value):
        return "numeric"
    if CLAUSE_RE.match(value):
        return "clause"
    if POINT_RE.match(value):
        return "point"
    if ROMAN_HEADING_RE.match(value):
        return "roman"
    if BULLET_RE.match(value):
        return "bullet"
    if value.startswith(PREAMBLE_PREFIXES):
        return "preamble"
    if re.match(r"^(Số\s*:|Hà Nội,?\s+ngày\b)", value, re.IGNORECASE):
        return "metadata"
    if re.match(r"^(Lời nói đầu|LỜI NÓI ĐẦU|Lời giới thiệu)$", value):
        return "heading"
    return ""


def should_end_paragraph(buffer: str) -> bool:
    return bool(re.search(r"[.;:!?…]$|[”’)]$", buffer.rstrip()))


def reconstruct_paragraphs(raw_passage: str) -> tuple[list[str], int]:
    value = strip_html(raw_passage)
    value = unicodedata.normalize("NFC", value)
    value = ZERO_WIDTH_RE.sub("", value)
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00a0", " ")

    lines = [normalize_inline(item) for item in value.split("\n")]
    lines = [item for item in lines if item]

    paragraphs: list[str] = []
    buffer = ""
    buffer_kind = ""
    omission_markers = 0

    def flush() -> None:
        nonlocal buffer, buffer_kind
        cleaned = normalize_inline(buffer)
        if cleaned:
            paragraphs.append(cleaned)
        buffer = ""
        buffer_kind = ""

    for line in lines:
        line_kind = classify_structural_line(line)
        if line_kind == "ellipsis":
            omission_markers += 1
            flush()
            continue

        if line_kind:
            flush()
            buffer = line
            buffer_kind = line_kind
            continue

        if not buffer:
            buffer = line
            buffer_kind = ""
            continue

        if buffer_kind:
            first_letter = next(
                (character for character in line if character.isalpha()),
                "",
            )
            if (
                first_letter
                and first_letter.islower()
                and buffer_kind
                in {
                    "metadata",
                    "chapter",
                    "section",
                    "article",
                    "numeric",
                    "clause",
                    "point",
                }
                and not should_end_paragraph(buffer)
            ):
                buffer = f"{buffer} {line}"
                continue
            flush()
            buffer = line
            buffer_kind = ""
            continue

        if should_end_paragraph(buffer):
            flush()
            buffer = line
            continue

        buffer = f"{buffer} {line}"

    flush()
    return paragraphs, omission_markers


def passage_from_paragraphs(paragraphs: Sequence[str]) -> str:
    return "\n\n".join(normalize_inline(item) for item in paragraphs if item)


def slug_title(name: str, link: str) -> str:
    candidate = normalize_inline(name)
    if not candidate and link:
        parsed = urlparse(link)
        candidate = Path(unquote(parsed.path)).stem
    if not candidate:
        return ""

    candidate = re.sub(r"-\d{5,}$", "", candidate)
    candidate = candidate.replace("_", " ").replace("-", " ")
    candidate = re.sub(r"\s+", " ", candidate).strip()
    return candidate


def extract_document_type(paragraphs: Sequence[str]) -> str:
    for paragraph in paragraphs[:80]:
        upper = normalize_inline(paragraph).upper().rstrip(":")
        if upper in DOCUMENT_TYPE_LINES:
            return upper[:1] + upper[1:].lower()
        if upper.startswith("TIÊU CHUẨN QUỐC GIA"):
            return "Tiêu chuẩn quốc gia"
        if upper.startswith("QUY CHUẨN KỸ THUẬT QUỐC GIA"):
            return "Quy chuẩn kỹ thuật quốc gia"
    return ""


def extract_document_number(paragraphs: Sequence[str]) -> str:
    head = "\n".join(paragraphs[:100])
    for pattern in DOCUMENT_NUMBER_RES:
        match = pattern.search(head)
        if match:
            return normalize_inline(match.group("number"))
    return ""


def extract_promulgation_date(paragraphs: Sequence[str]) -> str:
    head = "\n".join(paragraphs[:100])
    match = DATE_RE.search(head)
    if not match:
        return ""
    try:
        value = dt.date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return ""
    return value.isoformat()


def _iso_date(day: str, month: str, year: str) -> str:
    try:
        return dt.date(int(year), int(month), int(day)).isoformat()
    except (TypeError, ValueError):
        return ""


def extract_effective_date(paragraphs: Sequence[str]) -> str:
    text = "\n".join(paragraphs[:400])
    patterns = (
        re.compile(
            r"có\s+hiệu\s+lực(?:\s+thi\s+hành)?(?:\s+kể\s+từ)?\s+"
            r"ngày\s+(?P<day>\d{1,2})\s+tháng\s+(?P<month>\d{1,2})\s+"
            r"năm\s+(?P<year>\d{4})",
            re.IGNORECASE,
        ),
        re.compile(
            r"có\s+hiệu\s+lực(?:\s+thi\s+hành)?(?:\s+kể\s+từ)?\s+"
            r"ngày\s+(?P<day>\d{1,2})[./-](?P<month>\d{1,2})[./-]"
            r"(?P<year>\d{4})",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return _iso_date(
                match.group("day"),
                match.group("month"),
                match.group("year"),
            )
    return ""


def extract_issuing_authority(paragraphs: Sequence[str]) -> str:
    authority_re = re.compile(
        r"^(?:QUỐC HỘI|CHÍNH PHỦ|THỦ TƯỚNG CHÍNH PHỦ|"
        r"CHỦ TỊCH NƯỚC|TÒA ÁN NHÂN DÂN TỐI CAO|"
        r"VIỆN KIỂM SÁT NHÂN DÂN TỐI CAO|KIỂM TOÁN NHÀ NƯỚC|"
        r"BỘ(?:\s+[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s,&/-]*)?|"
        r"HỘI ĐỒNG NHÂN DÂN(?:\s+[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s,&/-]*)?|"
        r"ỦY BAN NHÂN DÂN(?:\s+[A-ZÀ-ỸĐ][A-ZÀ-ỸĐ\s,&/-]*)?)$"
    )
    ignored = {
        "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM",
        "ĐỘC LẬP - TỰ DO - HẠNH PHÚC",
    }
    for paragraph in paragraphs[:60]:
        candidates = re.split(
            r"-{2,}|\bCỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\b",
            normalize_inline(paragraph).upper(),
            maxsplit=2,
        )
        for raw_candidate in candidates:
            candidate = raw_candidate.strip(" :-–—")
            if candidate in ignored or len(candidate) > 160:
                continue
            if authority_re.fullmatch(candidate):
                return candidate[:1] + candidate[1:].lower()
    return ""


def extract_source_domain(link: str) -> str:
    if not link:
        return ""
    try:
        return (urlparse(link).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def clean_title_candidate(value: str) -> str:
    text = normalize_inline(value)
    text = re.sub(r"^\W+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -–—:")


def extract_official_title(
    paragraphs: Sequence[str],
    document_type: str,
    fallback_name: str,
    link: str,
) -> tuple[str, str]:
    for index, paragraph in enumerate(paragraphs[:120]):
        upper = paragraph.upper().rstrip(":")
        if upper not in DOCUMENT_TYPE_LINES:
            continue
        candidates: list[str] = []
        for following in paragraphs[index + 1 : index + 8]:
            if following.startswith(PREAMBLE_PREFIXES):
                break
            if ARTICLE_RE.match(following) or CHAPTER_RE.match(following):
                break
            if re.match(r"^(Số\s*:|Hà Nội,?\s+ngày\b)", following, re.I):
                continue
            if len(following) > 400:
                break
            candidates.append(following)
            if not is_all_caps_heading(following):
                break
        candidate = clean_title_candidate(" ".join(candidates))
        if len(candidate) >= 12:
            if document_type and not candidate.lower().startswith(
                document_type.lower()
            ):
                return f"{document_type}: {candidate}", "passage"
            return candidate, "passage"


    for paragraph in paragraphs[:40]:
        if not paragraph.upper().startswith("TIÊU CHUẨN QUỐC GIA"):
            continue
        code_matches = list(
            re.finditer(r"\b\d{3,7}:\d{4}\b", paragraph)
        )
        if not code_matches:
            continue
        candidate = paragraph[code_matches[-1].end() :].strip(" -–—:")
        for english_marker in (
            " Conformity assessment",
            " Requirements for",
            " Specification for",
            " Test methods",
        ):
            marker_index = candidate.find(english_marker)
            if marker_index > 0:
                candidate = candidate[:marker_index]
                break
        candidate = clean_title_candidate(candidate)
        if len(candidate) >= 12:
            return candidate, "passage"


    for paragraph in paragraphs[:120]:
        upper = paragraph.upper()
        if (
            len(paragraph) >= 20
            and is_all_caps_heading(paragraph)
            and any(
                marker in upper
                for marker in (
                    "YÊU CẦU",
                    "QUY ĐỊNH",
                    "HƯỚNG DẪN",
                    "ĐÁNH GIÁ",
                    "AN TOÀN",
                )
            )
        ):
            return clean_title_candidate(paragraph), "passage"

    fallback = clean_title_candidate(slug_title(fallback_name, link))
    if fallback:
        return fallback, "slug"
    if document_type:
        return document_type, "document_type"
    return "", "missing"


def load_external_metadata(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    records: list[dict[str, Any]] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid metadata JSONL line {line_number}: {path}"
                    ) from exc
                if isinstance(value, dict):
                    records.append(value)
    else:
        root = load_json(path)
        if isinstance(root, list):
            records = [item for item in root if isinstance(item, dict)]
        elif isinstance(root, dict):
            if isinstance(root.get("documents"), dict):
                for key, value in root["documents"].items():
                    if isinstance(value, dict):
                        records.append({"context_id": key, **value})
            else:
                for key, value in root.items():
                    if isinstance(value, dict):
                        records.append({"context_id": key, **value})
    result: dict[str, dict[str, Any]] = {}
    for record in records:
        context_id = normalize_id(
            record.get("context_id", record.get("id"))
        )
        if context_id:
            result[context_id] = record
    return result


def load_context_object(path: Path) -> Mapping[str, Any]:
    root = load_json(path)
    if not isinstance(root, dict):
        raise ValueError("Context JSON must be an object")
    return root


def find_administrative_restart(
    value: str,
) -> tuple[Optional[re.Match[str]], str]:
    anchored = ADMIN_RESTART_RE.search(value)
    if anchored is not None:
        return anchored, "anchored"

    for candidate in ADMIN_INLINE_RESTART_RE.finditer(value):
        prefix = value[: candidate.start()]
        suffix = value[candidate.start() : candidate.start() + 2500]
        if (
            ADMIN_SIGNATURE_CUE_RE.search(prefix) is not None
            or ADMIN_ATTACHMENT_CUE_RE.search(suffix) is not None
        ):
            return candidate, "inline_guarded"
    return None, ""


def administrative_tail_has_legal_signal(values: Sequence[str]) -> bool:
    strong_signals = 0
    numbered_provisions = 0
    for item in values:
        value = normalize_inline(item)
        if not value:
            continue
        restart, _ = find_administrative_restart(value)
        if restart is not None:
            return True
        kind = classify_structural_line(value)
        if kind in {
            "document_type",
            "chapter",
            "section",
            "article",
            "appendix",
            "numeric",
            "roman",
            "heading",
        }:
            strong_signals += 1
        elif kind == "clause" and len(value) >= 40:
            numbered_provisions += 1
        if ADMIN_ATTACHMENT_CUE_RE.search(value):
            strong_signals += 1
        if (
            len(value) >= 12
            and is_all_caps_heading(value)
            and ADMIN_SIGNATURE_CUE_RE.fullmatch(value) is None
        ):
            strong_signals += 1
    return strong_signals > 0 or numbered_provisions >= 2


def remove_administrative_blocks(
    paragraphs: Sequence[str],
) -> tuple[list[str], dict[str, Any]]:
    values = [str(item or "").strip() for item in paragraphs if str(item or "").strip()]
    output: list[str] = []
    removed_chars = 0
    removed_segments = 0
    restart_count = 0
    inline_restart_count = 0
    ambiguous_blocks_preserved = 0
    ambiguous_chars_preserved = 0
    removed_paragraphs: list[str] = []
    input_chars = sum(len(item) for item in values)

    def remember_removed(*items: str) -> None:
        for item in items:
            normalized = str(item or "").strip()
            if normalized:
                removed_paragraphs.append(normalized)

    index = 0
    while index < len(values):
        value = values[index]
        start = ADMIN_START_RE.search(value)
        if start is None:
            output.append(value)
            index += 1
            continue

        legal_prefix = value[: start.start()].strip()
        admin_tail = value[start.start() :]
        restart, strategy = find_administrative_restart(admin_tail)
        if restart is not None:
            if legal_prefix:
                output.append(legal_prefix)
            removed_part = admin_tail[: restart.start()]
            remember_removed(removed_part)
            removed_chars += len(removed_part)
            removed_segments += 1
            resumed = admin_tail[restart.start() :].strip()
            if resumed:
                output.append(resumed)
            restart_count += 1
            inline_restart_count += int(strategy == "inline_guarded")
            index += 1
            continue

        restart_index: Optional[int] = None
        restart_match: Optional[re.Match[str]] = None
        restart_strategy = ""
        for following_index in range(index + 1, len(values)):
            candidate, candidate_strategy = find_administrative_restart(
                values[following_index]
            )
            if candidate is not None:
                restart_index = following_index
                restart_match = candidate
                restart_strategy = candidate_strategy
                break

        if restart_index is not None and restart_match is not None:
            if legal_prefix:
                output.append(legal_prefix)
            remember_removed(admin_tail)
            removed_chars += len(admin_tail)
            removed_segments += 1
            for removed_value in values[index + 1 : restart_index]:
                remember_removed(removed_value)
                removed_chars += len(removed_value)
                removed_segments += 1
            restart_value = values[restart_index]
            removed_prefix = restart_value[: restart_match.start()]
            remember_removed(removed_prefix)
            removed_chars += len(removed_prefix)
            if removed_prefix.strip():
                removed_segments += 1
            resumed = restart_value[restart_match.start() :].strip()
            if resumed:
                output.append(resumed)
            restart_count += 1
            inline_restart_count += int(
                restart_strategy == "inline_guarded"
            )
            index = restart_index + 1
            continue

        uncertain_tail = [admin_tail, *values[index + 1 :]]
        uncertain_chars = sum(len(item) for item in uncertain_tail)
        uncertain_fraction_ppm = round(
            1_000_000 * uncertain_chars / max(1, input_chars)
        )
        preserve_tail = (
            uncertain_fraction_ppm >= MAX_UNCERTAIN_ADMIN_REMOVAL_PPM
            or administrative_tail_has_legal_signal(uncertain_tail)
        )
        if preserve_tail:


            output.extend(values[index:])
            ambiguous_blocks_preserved += 1
            ambiguous_chars_preserved += uncertain_chars
            break

        if legal_prefix:
            output.append(legal_prefix)
        remember_removed(*uncertain_tail)
        removed_chars += uncertain_chars
        removed_segments += len(uncertain_tail)
        break

    return output, {
        "removed_chars": removed_chars,
        "removed_segments": removed_segments,
        "restart_count": restart_count,
        "inline_restart_count": inline_restart_count,
        "ambiguous_blocks_preserved": ambiguous_blocks_preserved,
        "ambiguous_chars_preserved": ambiguous_chars_preserved,
        "input_chars": input_chars,
        "removed_fraction_ppm": round(
            1_000_000 * removed_chars / max(1, input_chars)
        ),


        "_removed_paragraphs": removed_paragraphs,
    }


def parse_context_file(
    path: Path,
    external_metadata: Mapping[str, Mapping[str, Any]],
) -> ParsedContext:
    anomalies: list[dict[str, Any]] = []
    try:
        root = load_context_object(path)
    except Exception as exc:
        return ParsedContext(
            document=None,
            units=[],
            anomalies=[
                {
                    "severity": "error",
                    "code": "invalid_context_json",
                    "source_file": str(path),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
        )

    context_id = normalize_id(
        root.get("id", root.get("context_id", root.get("doc_id")))
    )
    filename_match = re.search(r"context[_-](\d+)", path.stem, re.I)
    filename_id = filename_match.group(1) if filename_match else ""
    if not context_id:
        context_id = filename_id
    if not context_id:
        return ParsedContext(
            document=None,
            units=[],
            anomalies=[
                {
                    "severity": "error",
                    "code": "missing_context_id",
                    "source_file": str(path),
                }
            ],
        )
    if filename_id and filename_id != context_id:
        anomalies.append(
            {
                "severity": "warning",
                "code": "filename_id_mismatch",
                "context_id": context_id,
                "filename_id": filename_id,
                "source_file": str(path),
            }
        )

    passage = str(
        root.get(
            "passage",
            root.get("content", root.get("text", root.get("document", ""))),
        )
        or ""
    )
    if not normalize_inline(passage):
        return ParsedContext(
            document=None,
            units=[],
            anomalies=[
                {
                    "severity": "error",
                    "code": "empty_passage",
                    "context_id": context_id,
                    "source_file": str(path),
                }
            ],
        )

    link = normalize_inline(root.get("link", root.get("url", "")))
    name = normalize_inline(root.get("name", root.get("title", "")))
    paragraphs, omission_markers = reconstruct_paragraphs(passage)
    paragraphs, administrative_stats = remove_administrative_blocks(paragraphs)
    administrative_recovery_paragraphs = list(
        administrative_stats.pop("_removed_paragraphs", [])
    )
    administrative_stats["recovery_segments"] = len(
        administrative_recovery_paragraphs
    )
    administrative_stats["recovery_source_chars"] = sum(
        len(value) for value in administrative_recovery_paragraphs
    )
    normalized_passage = passage_from_paragraphs(paragraphs)
    document_type = extract_document_type(paragraphs)
    document_number = extract_document_number(paragraphs)
    promulgation_date = extract_promulgation_date(paragraphs)
    effective_date = extract_effective_date(paragraphs)
    issuing_authority = extract_issuing_authority(paragraphs)
    source_domain = extract_source_domain(link)
    title, title_source = extract_official_title(
        paragraphs,
        document_type,
        name,
        link,
    )

    metadata = dict(external_metadata.get(context_id, {}))
    official_metadata_title = normalize_inline(
        metadata.get("official_title", metadata.get("title", ""))
    )
    if official_metadata_title:
        title = official_metadata_title
        title_source = "metadata_file"

    document = SourceDocument(
        context_id=context_id,
        source_file=str(path),
        link=link,
        name=name,
        raw_passage=passage,
        normalized_passage=normalized_passage,
        paragraphs=paragraphs,
        title=title,
        title_source=title_source,
        document_type=document_type,
        document_number=document_number,
        promulgation_date=promulgation_date,
        effective_date=effective_date,
        issuing_authority=issuing_authority,
        source_domain=source_domain,
        content_sha256=sha256_text(normalized_passage),
        omission_markers=omission_markers,
        administrative_recovery_paragraphs=(
            administrative_recovery_paragraphs
        ),
        metadata=metadata,
    )
    units = parse_legal_units(document)

    if title_source in {"missing", "document_type"}:
        anomalies.append(
            {
                "severity": "warning",
                "code": "weak_or_missing_title",
                "context_id": context_id,
                "title_source": title_source,
                "source_file": str(path),
            }
        )
    if omission_markers:
        anomalies.append(
            {
                "severity": "info",
                "code": "omission_markers_removed",
                "context_id": context_id,
                "count": omission_markers,
            }
        )
    if administrative_stats["removed_segments"]:
        anomalies.append(
            {
                "severity": "info",
                "code": "administrative_tail_removed",
                "context_id": context_id,
                **administrative_stats,
            }
        )


        if administrative_stats["removed_fraction_ppm"] >= 350_000:
            anomalies.append(
                {
                    "severity": "warning",
                    "code": "large_administrative_removal",
                    "context_id": context_id,
                    **administrative_stats,
                }
            )
    if administrative_stats["ambiguous_blocks_preserved"]:
        anomalies.append(
            {
                "severity": "warning",
                "code": "ambiguous_administrative_tail_preserved",
                "context_id": context_id,
                **administrative_stats,
            }
        )
    if not units:
        anomalies.append(
            {
                "severity": "error",
                "code": "no_legal_units",
                "context_id": context_id,
                "source_file": str(path),
            }
        )
    return ParsedContext(document=document, units=units, anomalies=anomalies)


@dataclass(slots=True)
class _HierarchyState:
    part: str = ""
    chapter: str = ""
    section: str = ""
    appendix: str = ""
    article: str = ""
    article_label: str = ""
    clause: str = ""
    clause_label: str = ""
    point: str = ""
    point_label: str = ""
    numeric_path: list[str] = field(default_factory=list)

    def clear_below(self, level: str) -> None:
        if level == "part":
            self.chapter = ""
            self.section = ""
            self.appendix = ""
        if level in {"part", "chapter"}:
            self.section = ""
        if level in {"part", "chapter", "section", "appendix"}:
            self.article = ""
            self.article_label = ""
            self.numeric_path = []
        if level in {
            "part",
            "chapter",
            "section",
            "appendix",
            "article",
            "numeric",
        }:
            self.clause = ""
            self.clause_label = ""
        if level in {
            "part",
            "chapter",
            "section",
            "appendix",
            "article",
            "numeric",
            "clause",
        }:
            self.point = ""
            self.point_label = ""

    def path(self) -> list[str]:
        values = [
            self.part,
            self.chapter,
            self.section,
            self.appendix,
            self.article,
            *self.numeric_path,
            self.clause,
            self.point,
        ]
        output: list[str] = []
        for value in values:
            cleaned = normalize_inline(value)
            if cleaned and cleaned not in output:
                output.append(cleaned)
        return output


def heading_label(kind: str, label: str, title: str) -> str:
    head = normalize_inline(f"{kind} {label}".strip())
    tail = normalize_inline(title)
    return f"{head}. {tail}".strip(". ") if tail else head


def compact_article_title(value: str) -> str:
    title = normalize_inline(value)
    lower = title.lower()
    body_markers = (
        "ban hành kèm theo",
        "có hiệu lực",
        "chịu trách nhiệm",
        "quy định rằng",
        "giao ",
    )
    if (
        len(title) > 150
        or title.endswith((".", ";"))
        or any(marker in lower for marker in body_markers)
    ):
        return ""
    return title


def compact_numeric_title(value: str) -> str:
    title = normalize_inline(value)
    for marker in (
        " Tiêu chuẩn này ",
        " Quy chuẩn này ",
        " Việc ",
        " Tổ chức ",
        " CHÚ THÍCH",
    ):
        index = title.find(marker)
        if 2 <= index <= 160:
            title = title[:index]
            break
    if len(title) > 180:
        title = title[:177].rstrip() + "…"
    return title


def compact_clause_heading(value: str) -> str:
    raw = normalize_inline(value)
    title = raw.rstrip(":")
    if not title or len(title) > 180:
        return ""
    if raw.endswith(":") or (
        len(title.split()) <= 14
        and not title.endswith((".", ";", "?", "!"))
    ):
        return title
    return ""


def suppress_redundant_heading_units(units: Sequence[LegalUnit]) -> int:
    suppressed = 0
    for index, unit in enumerate(units[:-1]):
        if not unit.indexable:
            continue
        following = units[index + 1]
        article_heading = (
            unit.kind == "article"
            and bool(unit.article_label)
            and following.article_label == unit.article_label
            and following.kind in {"clause", "point", "bullet"}
            and bool(compact_article_title(unit.title))
            and len(unit.text) <= 180
        )
        clause_heading = (
            unit.kind == "clause"
            and bool(unit.clause_label)
            and following.article_label == unit.article_label
            and following.clause_label == unit.clause_label
            and following.kind in {"point", "bullet"}
            and bool(compact_clause_heading(unit.title or unit.text))
            and len(unit.text) <= 180
        )
        if article_heading or clause_heading:
            unit.indexable = False
            unit.excluded_reason = "hierarchy_heading"
            suppressed += 1
    return suppressed


def is_probable_heading_title(paragraph: str) -> bool:
    value = normalize_inline(paragraph)
    if not value or len(value) > 220:
        return False
    if classify_structural_line(value):
        return False
    if should_end_paragraph(value) and not is_all_caps_heading(value):
        return False
    return is_all_caps_heading(value) or len(value.split()) <= 16


def parse_legal_units(document: SourceDocument) -> list[LegalUnit]:
    paragraphs = document.paragraphs
    has_article_headings = any(ARTICLE_RE.match(item) for item in paragraphs)
    state = _HierarchyState()
    units: list[LegalUnit] = []
    current: Optional[dict[str, Any]] = None
    seen_primary_structure = False
    trailing_toc = False
    pending_hierarchy_field = ""

    def flush() -> None:
        nonlocal current
        if not current:
            return
        text = "\n".join(current["paragraphs"]).strip()
        if text:
            unit_order = len(units)
            units.append(
                LegalUnit(
                    unit_id=f"ctx{document.context_id}__u{unit_order:05d}",
                    context_id=document.context_id,
                    order=unit_order,
                    kind=current["kind"],
                    label=current.get("label", ""),
                    title=current.get("title", ""),
                    text=text,
                    hierarchy=list(current["hierarchy"]),
                    article_label=current.get("article_label", ""),
                    clause_label=current.get("clause_label", ""),
                    point_label=current.get("point_label", ""),
                    indexable=bool(current.get("indexable", True)),
                    excluded_reason=current.get("excluded_reason", ""),
                )
            )
        current = None

    def start_unit(
        kind: str,
        paragraph: str,
        label: str = "",
        title: str = "",
        indexable: bool = True,
        excluded_reason: str = "",
    ) -> None:
        nonlocal current
        flush()
        current = {
            "kind": kind,
            "label": label,
            "title": title,
            "paragraphs": [paragraph],
            "hierarchy": state.path(),
            "article_label": state.article_label,
            "clause_label": state.clause_label,
            "point_label": state.point_label,
            "indexable": indexable,
            "excluded_reason": excluded_reason,
        }

    for position, paragraph in enumerate(paragraphs):
        value = normalize_inline(paragraph)
        upper = value.upper().rstrip(":")

        if upper in NON_CONTENT_HEADINGS and position > len(paragraphs) * 0.55:
            flush()
            trailing_toc = True
            continue
        if trailing_toc:
            start_unit(
                "trailing_toc",
                value,
                indexable=False,
                excluded_reason="trailing_toc",
            )
            continue

        if pending_hierarchy_field and is_probable_heading_title(value):
            existing = getattr(state, pending_hierarchy_field)
            setattr(
                state,
                pending_hierarchy_field,
                f"{existing}. {value}".strip(". "),
            )
            pending_hierarchy_field = ""
            continue
        pending_hierarchy_field = ""

        match = CHAPTER_RE.match(value)
        if match:
            flush()
            kind = match.group("kind").title()
            label = match.group("label")
            title = match.group("title")
            field_name = "part" if kind.lower() == "phần" else "chapter"
            state.clear_below(field_name)
            setattr(state, field_name, heading_label(kind, label, title))
            if not title:
                pending_hierarchy_field = field_name
            seen_primary_structure = True
            continue

        match = SECTION_RE.match(value)
        if match:
            flush()
            kind = match.group("kind").title()
            label = match.group("label")
            title = match.group("title")
            state.clear_below("section")
            state.section = heading_label(kind, label, title)
            if not title:
                pending_hierarchy_field = "section"
            seen_primary_structure = True
            continue

        match = APPENDIX_RE.match(value)
        if match:
            flush()
            kind = "Phụ lục"
            label = match.group("label")
            title = match.group("title")
            state.clear_below("appendix")
            state.appendix = heading_label(kind, label, title)
            if not title:
                pending_hierarchy_field = "appendix"
            seen_primary_structure = True
            continue

        match = ARTICLE_RE.match(value)
        if match:
            label = match.group("label")
            title = match.group("title")
            state.clear_below("article")
            state.article_label = label
            state.article = heading_label(
                "Điều",
                label,
                compact_article_title(title),
            )
            seen_primary_structure = True
            start_unit("article", value, label=label, title=title)
            continue

        numeric_match = NUMERIC_HEADING_RE.match(value)
        if numeric_match and not state.article_label:
            label = numeric_match.group("label")
            title = numeric_match.group("body")
            state.clear_below("numeric")
            segments = label.split(".")
            depth = len(segments)
            state.numeric_path = state.numeric_path[: max(0, depth - 1)]
            state.numeric_path.append(
                heading_label(
                    "",
                    label,
                    compact_numeric_title(title),
                ).strip()
            )
            seen_primary_structure = True
            start_unit(
                "numeric_section",
                value,
                label=label,
                title=title,
            )
            continue

        clause_match = CLAUSE_RE.match(value)
        if clause_match and not has_article_headings:
            label = clause_match.group("label")
            title = clause_match.group("body")
            state.clear_below("numeric")
            state.numeric_path = [
                heading_label(
                    "",
                    label,
                    compact_numeric_title(title),
                ).strip()
            ]
            seen_primary_structure = True
            start_unit(
                "numeric_section",
                value,
                label=label,
                title=title,
            )
            continue

        if clause_match and (state.article_label or seen_primary_structure):
            label = clause_match.group("label")
            body = clause_match.group("body")
            state.clear_below("clause")
            state.clause_label = label
            state.clause = heading_label(
                "Khoản",
                label,
                compact_clause_heading(body),
            )
            start_unit("clause", value, label=label, title=body[:160])
            continue

        point_match = POINT_RE.match(value)
        if point_match and (
            state.article_label or state.clause_label or seen_primary_structure
        ):
            label = point_match.group("label").lower()
            body = point_match.group("body")
            state.clear_below("point")
            state.point_label = label
            state.point = heading_label("Điểm", label, "")
            start_unit("point", value, label=label, title=body[:160])
            continue

        roman_match = ROMAN_HEADING_RE.match(value)
        if roman_match and not state.article_label:
            label = roman_match.group("label")
            body = roman_match.group("body")
            state.clear_below("numeric")
            state.numeric_path = [
                heading_label("", label, body).strip()
            ]
            seen_primary_structure = True
            start_unit("roman_section", value, label=label, title=body)
            continue

        bullet_match = BULLET_RE.match(value)
        if bullet_match and current:


            start_unit(
                "bullet",
                value,
                label=bullet_match.group("label"),
                title=bullet_match.group("body")[:160],
            )
            continue

        if current:
            current["paragraphs"].append(value)
        else:
            indexable = seen_primary_structure
            reason = "" if indexable else "preamble"
            start_unit(
                "paragraph" if indexable else "preamble",
                value,
                indexable=indexable,
                excluded_reason=reason,
            )

    flush()


    if not any(unit.indexable for unit in units):
        for unit in units:
            if unit.excluded_reason == "preamble":
                unit.indexable = True
                unit.excluded_reason = ""
                unit.kind = "paragraph"
    suppress_redundant_heading_units(units)
    return units


def common_hierarchy_path(units: Sequence[LegalUnit]) -> list[str]:
    if not units:
        return []
    paths = [unit.hierarchy for unit in units if unit.hierarchy]
    if not paths:
        return []
    limit = min(len(path) for path in paths)
    output: list[str] = []
    for index in range(limit):
        candidate = normalize_inline(paths[0][index])
        if candidate and all(
            normalize_inline(path[index]) == candidate for path in paths[1:]
        ):
            output.append(candidate)
        else:
            break
    return output


def retrieval_scope_key(unit: LegalUnit) -> tuple[str, ...]:
    path = [normalize_inline(item) for item in unit.hierarchy if normalize_inline(item)]
    if unit.clause_label and (
        unit.kind in {"point", "bullet"}
        or unit.excluded_reason == "hierarchy_heading"
    ):
        clause_pattern = re.compile(
            rf"(?i)^Khoản\s+{re.escape(unit.clause_label)}(?:\b|[.\s])"
        )
        for index in range(len(path) - 1, -1, -1):
            if clause_pattern.search(path[index]):
                return tuple(path[: index + 1])
    if unit.article_label:
        article_pattern = re.compile(
            rf"(?i)^Điều\s+{re.escape(unit.article_label)}(?:\b|[.\s])"
        )
        for index in range(len(path) - 1, -1, -1):
            if article_pattern.search(path[index]):
                return tuple(path[: index + 1])
    if len(path) > 1:
        return tuple(path[:-1])
    return tuple(path)


def hierarchy_prefix_for_path(
    document: SourceDocument,
    hierarchy: Sequence[str],
    max_chars: int,
) -> str:
    parts: list[str] = []
    if document.title:
        parts.append(f"[VĂN BẢN] {document.title}")
    if document.document_number:
        parts.append(f"[SỐ/KÝ HIỆU] {document.document_number}")
    cleaned_path = [normalize_inline(item) for item in hierarchy if normalize_inline(item)]
    if cleaned_path:
        parts.append(f"[PHÂN CẤP] {' > '.join(cleaned_path)}")

    generated_title = normalize_inline(
        document.metadata.get(
            "generated_title",
            document.metadata.get("llm_title", ""),
        )
    )
    if generated_title and document.title_source in {"missing", "document_type"}:
        parts.append(f"[TIÊU ĐỀ GỢI Ý] {generated_title}")

    prefix = "\n".join(parts)
    if len(prefix) <= max_chars:
        return prefix

    local_path = " > ".join(cleaned_path[-3:])
    compact_parts: list[str] = []
    if document.title:
        title_budget = max(80, max_chars // 2)
        title = document.title
        if len(title) > title_budget:
            title = title[: title_budget - 1].rstrip() + "…"
        compact_parts.append(f"[VĂN BẢN] {title}")
    if local_path:
        compact_parts.append(f"[PHÂN CẤP] {local_path}")
    compact = "\n".join(compact_parts)
    if len(compact) > max_chars:
        compact = compact[: max_chars - 1].rstrip() + "…"
    return compact


def build_packed_retrieval_chunks(
    document: SourceDocument,
    units: Sequence[LegalUnit],
    long_chunks: Sequence[Mapping[str, Any]],
    config: BuildConfig,
    segmenter: WordSegmenter,
) -> list[dict[str, Any]]:
    unit_to_long: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for long_chunk in long_chunks:
        for unit_id in long_chunk["unit_ids"]:
            unit_to_long[str(unit_id)].append(long_chunk)

    expanded: list[tuple[LegalUnit, str, int, tuple[str, ...]]] = []
    for unit in units:


        if not unit.indexable or not normalize_inline(unit.text):
            continue
        overlap = (
            config.retrieval_overlap_chars
            if len(normalize_inline(unit.text)) > config.retrieval_body_chars
            else 0
        )
        pieces = split_at_boundaries(
            unit.text,
            config.retrieval_body_chars,
            overlap_chars=overlap,
        )
        scope = retrieval_scope_key(unit)
        for piece_index, piece in enumerate(pieces):
            expanded.append((unit, piece, piece_index, scope))

    groups: list[list[tuple[LegalUnit, str, int, tuple[str, ...]]]] = []
    current: list[tuple[LegalUnit, str, int, tuple[str, ...]]] = []

    def group_body_length(
        items: Sequence[tuple[LegalUnit, str, int, tuple[str, ...]]],
    ) -> int:
        return len("\n".join(item[1] for item in items))

    for item in expanded:
        same_scope = bool(current) and current[-1][3] == item[3]
        candidate = [*current, item]
        if current and (
            not same_scope
            or group_body_length(candidate) > config.retrieval_body_chars
        ):
            groups.append(current)
            current = [item]
        else:
            current = candidate
    if current:
        groups.append(current)

    chunks: list[dict[str, Any]] = []
    for chunk_index, group in enumerate(groups):
        body = "\n".join(item[1] for item in group)
        group_units = list(dict.fromkeys(item[0].unit_id for item in group))
        unique_units: list[LegalUnit] = []
        seen_units: set[str] = set()
        for item in group:
            if item[0].unit_id not in seen_units:
                seen_units.add(item[0].unit_id)
                unique_units.append(item[0])
        representative = unique_units[len(unique_units) // 2]
        shared_path = common_hierarchy_path(unique_units)
        prefix = hierarchy_prefix_for_path(
            document,
            shared_path or representative.hierarchy,
            config.max_prefix_chars,
        )
        dense_text = (
            f"{prefix}\n[NỘI DUNG]\n{body}".strip() if prefix else body
        )
        parent = choose_parent_long_chunk(
            representative,
            unit_to_long.get(representative.unit_id, []),
        )
        parent_long_id = str(parent["chunk_id"]) if parent else ""
        chunk_id = f"ctx{document.context_id}__p{chunk_index:05d}"
        record = {
            "context_id": document.context_id,
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "view": "packed_retrieval",
            "unit_ids": group_units,
            "unit_order_start": min(item[0].order for item in group),
            "unit_order_end": max(item[0].order for item in group),
            "hierarchy": shared_path or representative.hierarchy,
            "parent_long_chunk_id": parent_long_id,
            "raw_body": body,
            "content_Article": dense_text,
            "dense_text": dense_text,
            "lexical_text": segmenter.segment(dense_text),
            "body_char_count": len(body),
            "full_char_count": len(dense_text),
            "content_sha256": sha256_text(dense_text),
        }
        chunks.append(record)
    return chunks


def split_at_boundaries(
    text: str,
    max_chars: int,
    overlap_chars: int = 0,
) -> list[str]:
    value = normalize_inline(text.replace("\n", " \n "))
    value = re.sub(r"\s*\n\s*", "\n", value)
    if len(value) <= max_chars:
        return [value] if value else []

    pieces: list[str] = []
    start = 0
    length = len(value)
    minimum_cut = max(1, int(max_chars * 0.55))

    while start < length:
        end = min(length, start + max_chars)
        if end < length:
            window = value[start:end]
            cut = -1
            for pattern in (
                r"\n",
                r"(?<=[.!?…])\s+",
                r"(?<=[;:])\s+",
                r"\s+",
            ):
                matches = list(re.finditer(pattern, window))
                eligible = [
                    match
                    for match in matches
                    if match.end() >= minimum_cut
                ]
                if eligible:
                    cut = eligible[-1].end()
                    break
            if cut <= 0:
                cut = len(window)
            end = start + cut

        piece = normalize_inline(value[start:end].replace("\n", " "))
        if piece:
            pieces.append(piece)
        if end >= length:
            break

        next_start = end
        if overlap_chars:
            proposed = max(start + 1, end - overlap_chars)
            overlap_window = value[proposed:end]


            clean_starts = list(
                re.finditer(r"(?<=[.!?…;:])\s+|\n+", overlap_window)
            )
            minimum_tail = min(32, max(8, overlap_chars // 4))
            clean_start = next(
                (
                    match.end()
                    for match in clean_starts
                    if len(overlap_window) - match.end() >= minimum_tail
                ),
                None,
            )
            if clean_start is not None:
                next_start = proposed + clean_start
            else:
                whitespace = value.find(" ", proposed, end)
                next_start = whitespace + 1 if whitespace >= 0 else proposed
        start = next_start

    return pieces


def build_long_chunks(
    document: SourceDocument,
    units: Sequence[LegalUnit],
    config: BuildConfig,
    segmenter: WordSegmenter,
) -> list[dict[str, Any]]:


    indexable_units = [
        unit
        for unit in units
        if unit.indexable and unit.text
    ]
    expanded: list[tuple[LegalUnit, str, int]] = []
    for unit in indexable_units:
        pieces = split_at_boundaries(
            unit.text,
            config.long_body_chars,
            config.long_overlap_chars,
        )
        for piece_index, piece in enumerate(pieces):
            expanded.append((unit, piece, piece_index))

    groups: list[list[tuple[LegalUnit, str, int]]] = []
    current: list[tuple[LegalUnit, str, int]] = []

    def body_length(
        items: Sequence[tuple[LegalUnit, str, int]],
    ) -> int:
        return len("\n\n".join(item[1] for item in items))

    def same_scope(
        left: tuple[LegalUnit, str, int],
        right: tuple[LegalUnit, str, int],
    ) -> bool:
        left_unit, _, _ = left
        right_unit, _, _ = right
        if left_unit.article_label or right_unit.article_label:
            return left_unit.article_label == right_unit.article_label
        left_parent = left_unit.hierarchy[:-1]
        right_parent = right_unit.hierarchy[:-1]
        return left_parent == right_parent

    for item in expanded:
        candidate = [*current, item]
        scope_break = current and not same_scope(current[-1], item)
        if current and (
            body_length(candidate) > config.long_body_chars or scope_break
        ):
            groups.append(current)
            overlap: list[tuple[LegalUnit, str, int]] = []
            if not scope_break and config.long_overlap_units:
                overlap = current[-config.long_overlap_units :]
                while (
                    overlap
                    and body_length([*overlap, item])
                    > config.long_body_chars
                ):
                    overlap.pop(0)
            current = [*overlap, item]
        else:
            current = candidate
    if current:
        groups.append(current)

    chunks: list[dict[str, Any]] = []
    for chunk_index, group in enumerate(groups):
        body = "\n\n".join(item[1] for item in group)
        unit_ids = list(dict.fromkeys(item[0].unit_id for item in group))
        unique_units: list[LegalUnit] = []
        seen_units: set[str] = set()
        for item in group:
            if item[0].unit_id not in seen_units:
                seen_units.add(item[0].unit_id)
                unique_units.append(item[0])
        representative = unique_units[len(unique_units) // 2]
        shared_path = common_hierarchy_path(unique_units)
        prefix = hierarchy_prefix_for_path(
            document,
            shared_path or representative.hierarchy,
            config.max_prefix_chars,
        )
        rerank_text = (
            f"{prefix}\n[NỘI DUNG]\n{body}".strip()
            if prefix
            else body
        )
        chunk_id = f"ctx{document.context_id}__l{chunk_index:05d}"
        chunks.append(
            {
                "context_id": document.context_id,
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "view": "long_reranking",
                "unit_ids": unit_ids,
                "unit_order_start": min(item[0].order for item in group),
                "unit_order_end": max(item[0].order for item in group),
                "hierarchy": shared_path or representative.hierarchy,
                "raw_body": body,
                "content_Article": rerank_text,
                "rerank_text": rerank_text,
                "lexical_text": segmenter.segment(rerank_text),
                "body_char_count": len(body),
                "full_char_count": len(rerank_text),
                "content_sha256": sha256_text(rerank_text),
            }
        )
    return chunks


def choose_parent_long_chunk(
    unit: LegalUnit,
    candidates: Sequence[Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    if not candidates:
        return None

    def key(candidate: Mapping[str, Any]) -> tuple[float, int, str]:
        start = int(candidate["unit_order_start"])
        end = int(candidate["unit_order_end"])
        center = (start + end) / 2
        distance = abs(center - unit.order)
        coverage = end - start
        return (distance, -coverage, str(candidate["chunk_id"]))

    return min(candidates, key=key)


def document_metadata_view(
    document: SourceDocument,
    segmenter: WordSegmenter,
) -> dict[str, Any]:
    generated_title = normalize_inline(
        document.metadata.get(
            "generated_title",
            document.metadata.get("llm_title", ""),
        )
    )
    generated_summary = normalize_inline(
        document.metadata.get(
            "generated_summary",
            document.metadata.get(
                "summary",
                document.metadata.get("llm_summary", ""),
            ),
        )
    )
    keyword_value = document.metadata.get(
        "generated_keywords",
        document.metadata.get("keywords", []),
    )
    if isinstance(keyword_value, str):
        keywords = [
            normalize_inline(item)
            for item in re.split(r"[,;|]", keyword_value)
            if normalize_inline(item)
        ]
    elif isinstance(keyword_value, list):
        keywords = [
            normalize_inline(item)
            for item in keyword_value
            if normalize_inline(item)
        ]
    else:
        keywords = []

    parts = []
    if document.title:
        parts.append(f"[TIÊU ĐỀ CHÍNH THỨC] {document.title}")
    if document.document_number:
        parts.append(f"[SỐ/KÝ HIỆU] {document.document_number}")
    if document.issuing_authority:
        parts.append(f"[CƠ QUAN BAN HÀNH] {document.issuing_authority}")
    if document.promulgation_date:
        parts.append(f"[NGÀY BAN HÀNH] {document.promulgation_date}")
    if document.effective_date:
        parts.append(f"[NGÀY HIỆU LỰC] {document.effective_date}")
    if generated_title and generated_title != document.title:
        parts.append(f"[TIÊU ĐỀ SINH] {generated_title}")
    if generated_summary:
        parts.append(f"[MÔ TẢ] {generated_summary}")
    if keywords:
        parts.append(f"[TỪ KHÓA] {'; '.join(keywords)}")
    text = "\n".join(parts)
    return {
        "context_id": document.context_id,
        "title": document.title,
        "title_source": document.title_source,
        "document_type": document.document_type,
        "document_number": document.document_number,
        "promulgation_date": document.promulgation_date,
        "effective_date": document.effective_date,
        "issuing_authority": document.issuing_authority,
        "source_domain": document.source_domain,
        "language": "vi",
        "generated_title": generated_title,
        "generated_summary": generated_summary,
        "keywords": keywords,
        "dense_text": text,
        "lexical_text": segmenter.segment(text),
        "content_sha256": sha256_text(text),
    }


def retrieval_metadata_text(
    document: SourceDocument,
    units: Sequence[LegalUnit],
    segmenter: WordSegmenter,
    max_chars: int,
) -> str:
    base = document_metadata_view(document, segmenter)
    parts = [normalize_inline(base.get("dense_text", ""))]
    headings: list[str] = []
    seen: set[str] = set()
    for unit in units:
        if not unit.indexable or unit.kind not in {
            "article",
            "numeric_section",
            "roman_section",
        }:
            continue
        candidate = ""
        if unit.hierarchy:
            candidate = normalize_inline(unit.hierarchy[-1])
        if not candidate:
            candidate = normalize_inline(unit.title or unit.text)
        key = candidate.casefold()
        if (
            not candidate
            or key in seen
            or len(candidate) > 220
            or len(candidate.split()) > 32
        ):
            continue
        seen.add(key)
        headings.append(candidate)
        if len(headings) >= 30:
            break
    if headings:
        parts.append(f"[CHỦ ĐỀ/ĐIỀU MỤC] {'; '.join(headings)}")

    text = "\n".join(part for part in parts if part).strip()
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    cut = max(
        truncated.rfind("; "),
        truncated.rfind("\n"),
        truncated.rfind(". "),
    )
    if cut >= int(max_chars * 0.7):
        truncated = truncated[:cut]
    return truncated.rstrip(" ;,.\n") + "…"


def unit_record(unit: LegalUnit) -> dict[str, Any]:
    return dataclasses.asdict(unit)


def dataset_name_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in (
        "-with-answers",
        "_with_answers",
        "-answers",
        "_answers",
        "-official",
        "_official",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return re.sub(r"[^0-9A-Za-zÀ-ỹ_-]+", "_", stem).strip("_") or "queries"


def parse_query_root(
    root: Any,
    dataset: str,
    source_file: str,
    read_labels: bool = False,
) -> list[QueryRecord]:
    records: list[QueryRecord] = []
    if isinstance(root, dict):
        iterable = root.items()
    elif isinstance(root, list):
        iterable = enumerate(root)
    else:
        raise ValueError("Query file must contain an object or array")

    for outer_id, value in iterable:
        if not isinstance(value, dict):
            continue
        query_id = normalize_id(
            value.get("qid", value.get("id", outer_id))
        )
        question = normalize_inline(
            value.get("question", value.get("query", value.get("text", "")))
        )
        answer_values: list[Any] = []
        if read_labels:
            answer_value = value.get("answer")
            if isinstance(answer_value, (str, int, float)):
                answer_values = [answer_value]
            elif isinstance(answer_value, list):
                answer_values = answer_value
        answers = [
            item
            for item in dict.fromkeys(
                normalize_id(item) for item in answer_values
            )
            if item
        ]
        if not query_id or not question:
            continue
        records.append(
            QueryRecord(
                dataset=dataset,
                query_id=query_id,
                question=question,
                normalized_question=question,
                answer_context_ids=answers,
                source_file=source_file,
            )
        )
    records.sort(key=lambda item: numeric_sort_key(item.query_id))
    return records


def load_query_files(
    paths: Sequence[Path],
    allow_partial: bool,
    labeled_paths: Sequence[Path] = (),
) -> tuple[list[QueryRecord], list[dict[str, Any]]]:
    all_queries: list[QueryRecord] = []
    anomalies: list[dict[str, Any]] = []
    seen_dataset_query: set[tuple[str, str]] = set()
    labeled = {Path(path).resolve() for path in labeled_paths}

    for path in paths:
        try:
            root, recovered = load_json_resilient(path, allow_partial)
            dataset = dataset_name_from_path(path)
            records = parse_query_root(
                root,
                dataset,
                str(path),
                read_labels=path.resolve() in labeled,
            )
        except Exception as exc:
            anomalies.append(
                {
                    "severity": "error",
                    "code": "invalid_query_file",
                    "source_file": str(path),
                    "message": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        if recovered:
            anomalies.append(
                {
                    "severity": "warning",
                    "code": "partial_query_json_recovered",
                    "source_file": str(path),
                    "recovered_records": len(records),
                }
            )
        for record in records:
            key = (record.dataset, record.query_id)
            if key in seen_dataset_query:
                anomalies.append(
                    {
                        "severity": "error",
                        "code": "duplicate_query_id",
                        "dataset": record.dataset,
                        "query_id": record.query_id,
                    }
                )
                continue
            seen_dataset_query.add(key)
            all_queries.append(record)
    return all_queries, anomalies


def build_segmenter(config: BuildConfig) -> WordSegmenter:
    if config.word_segmenter == "underthesea":
        return UndertheseaWordSegmenter()
    if config.word_segmenter == "vncorenlp":
        return VnCoreNLPWordSegmenter(config.vncorenlp_dir)
    return SimpleWordSegmenter()


def discover_context_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Contexts directory not found: {directory}")
    files = [
        path
        for path in directory.rglob("*.json")
        if path.is_file() and not path.name.startswith(".")
    ]

    def path_key(path: Path) -> tuple[int, Any, str]:
        match = re.search(r"context[_-](\d+)", path.stem, re.I)
        if match:
            return (0, int(match.group(1)), str(path))
        return (1, path.stem, str(path))

    files.sort(key=path_key)
    if not files:
        raise FileNotFoundError(
            f"No JSON context files found under: {directory}"
        )
    return files


def bounded_thread_map(
    executor: concurrent.futures.ThreadPoolExecutor,
    function: Any,
    items: Iterable[Path],
    max_pending: int,
) -> Iterator[Any]:
    iterator = iter(items)
    pending: deque[concurrent.futures.Future[Any]] = deque()
    for _ in range(max(1, max_pending)):
        try:
            item = next(iterator)
        except StopIteration:
            break
        pending.append(executor.submit(function, item))

    while pending:
        future = pending.popleft()
        yield future.result()
        try:
            item = next(iterator)
        except StopIteration:
            continue
        pending.append(executor.submit(function, item))


def _query_output_groups(
    queries: Sequence[QueryRecord],
) -> dict[str, list[QueryRecord]]:
    grouped: dict[str, list[QueryRecord]] = defaultdict(list)
    for query in queries:
        grouped[query.dataset].append(query)
    for records in grouped.values():
        records.sort(key=lambda item: numeric_sort_key(item.query_id))
    return dict(sorted(grouped.items()))


def compact_raw_record(document: SourceDocument) -> dict[str, Any]:
    record: dict[str, Any] = {
        "id": document.context_id,
        "passage": document.raw_passage,
    }
    if document.link:
        record["link"] = document.link
    if document.name:
        record["name"] = document.name
    return record


def compact_retrieval_record(
    context_id: str,
    chunk_id: str,
    text: str,
) -> dict[str, str]:
    normalized_lines = [
        normalize_inline(line)
        for line in unicodedata.normalize("NFC", str(text or "")).splitlines()
        if normalize_inline(line)
    ]
    return {
        "chunk_id": str(chunk_id),
        "context_id": str(context_id),
        "text": "\n".join(normalized_lines),
    }


def administrative_recovery_records(
    document: SourceDocument,
    config: BuildConfig,
) -> list[dict[str, str]]:
    source = "\n".join(document.administrative_recovery_paragraphs).strip()
    if not source:
        return []
    prefix = hierarchy_prefix_for_path(
        document,
        [],
        config.max_prefix_chars,
    )
    pieces = split_at_boundaries(
        source,
        config.retrieval_body_chars,
        overlap_chars=config.retrieval_overlap_chars,
    )
    records: list[dict[str, str]] = []
    for index, body in enumerate(pieces):
        text = f"{prefix}\n[NỘI DUNG]\n{body}".strip() if prefix else body
        records.append(
            compact_retrieval_record(
                document.context_id,
                f"ctx{document.context_id}__r{index:05d}",
                text,
            )
        )
    return records


def compact_candidate_view(chunk_id: Any) -> str:
    value = normalize_id(chunk_id)
    if value.endswith("__metadata"):
        return "document_metadata"
    if re.search(r"__p\d+$", value):
        return "packed_retrieval"
    if re.search(r"__l\d+$", value):
        return "article_window"
    if re.search(r"__r\d+$", value):
        return "lossless_recovery"
    return "unknown"


def retrieval_content_body(text: Any) -> str:
    value = unicodedata.normalize("NFC", str(text or ""))
    marker = "[NỘI DUNG]"
    if marker not in value:
        return ""
    return value.split(marker, 1)[1].strip()


def is_heading_only_retrieval_candidate(
    chunk_id: Any,
    text: Any,
) -> bool:
    if compact_candidate_view(chunk_id) in {
        "document_metadata",
        "lossless_recovery",
    }:
        return False
    body = retrieval_content_body(text)
    if not body or "\n" in body:
        return False
    value = normalize_inline(body)
    structural = classify_structural_line(value)
    if structural in {
        "chapter",
        "section",
        "appendix",
        "document_type",
        "heading",
    }:
        return True
    article = ARTICLE_RE.match(value)
    if article:
        return bool(compact_article_title(article.group("title")))
    for pattern in (NUMERIC_HEADING_RE, ROMAN_HEADING_RE, CLAUSE_RE):
        match = pattern.match(value)
        if match and compact_clause_heading(match.group("body")):
            return True
    return False


def _serialized_build_config(config: BuildConfig) -> dict[str, Any]:
    value = dataclasses.asdict(config)
    for key, item in list(value.items()):
        if isinstance(item, Path):
            value[key] = str(item)
        elif isinstance(item, list):
            value[key] = [str(v) if isinstance(v, Path) else v for v in item]
    return value


def run_compact_build(config: BuildConfig) -> dict[str, Any]:
    config.validate()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = config.output_dir / "corpus"
    queries_dir = config.output_dir / "queries"
    qrels_dir = config.output_dir / "qrels"
    analysis_dir = config.output_dir / "analysis"
    for directory in (corpus_dir, queries_dir, qrels_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if config.analysis_level == "full":
        analysis_dir.mkdir(parents=True, exist_ok=True)

    raw_path = corpus_dir / "corpus_raw.jsonl"
    retrieval_path = corpus_dir / "retrieval_chunks.jsonl"
    context_files = discover_context_files(config.contexts_dir)
    segmenter = build_segmenter(config)
    external_metadata = load_external_metadata(config.metadata_jsonl)
    queries, query_anomalies = load_query_files(
        config.query_files,
        config.allow_partial_json,
        config.labeled_query_files,
    )
    relevant_context_ids = {
        context_id
        for query in queries
        for context_id in query.answer_context_ids
    }

    counts: Counter[str] = Counter()
    anomaly_counts: Counter[str] = Counter()
    excluded_reason_counts: Counter[str] = Counter()
    unit_kind_counts: Counter[str] = Counter()
    title_source_counts: Counter[str] = Counter()
    packed_body_lengths = LengthDistribution()
    packed_text_lengths = LengthDistribution()
    article_body_lengths = LengthDistribution()
    article_text_lengths = LengthDistribution()
    recovery_body_lengths = LengthDistribution()
    recovery_text_lengths = LengthDistribution()
    anomaly_examples: list[dict[str, Any]] = []
    error_examples: list[dict[str, Any]] = []
    seen_context_ids: set[str] = set()
    seen_chunk_ids: set[str] = set()
    context_ids_with_retrieval: set[str] = set()
    empty_context_ids: set[str] = set()

    def collect_anomaly(anomaly: Mapping[str, Any]) -> None:
        severity = str(anomaly.get("severity", "unknown"))
        code = str(anomaly.get("code", "unknown"))
        anomaly_counts[severity] += 1
        anomaly_counts[f"code::{code}"] += 1
        if code == "administrative_tail_removed":
            counts["administrative_removed_segments"] += int(
                anomaly.get("removed_segments", 0)
            )
            counts["administrative_removed_chars"] += int(
                anomaly.get("removed_chars", 0)
            )
            counts["administrative_restart_count"] += int(
                anomaly.get("restart_count", 0)
            )
            counts["administrative_recovery_segments"] += int(
                anomaly.get("recovery_segments", 0)
            )
            counts["administrative_recovery_source_chars"] += int(
                anomaly.get("recovery_source_chars", 0)
            )
        if len(anomaly_examples) < 100:
            anomaly_examples.append(dict(anomaly))
        if severity == "error" and len(error_examples) < 100:
            error_examples.append(dict(anomaly))

    for anomaly in query_anomalies:
        collect_anomaly(anomaly)

    analysis_stack = contextlib.ExitStack()
    with analysis_stack:
        unit_writer = None
        detailed_writer = None
        long_writer = None
        anomaly_writer = None
        if config.analysis_level == "full":
            unit_writer = analysis_stack.enter_context(
                AtomicJSONLWriter(analysis_dir / "legal_units.jsonl")
            )
            detailed_writer = analysis_stack.enter_context(
                AtomicJSONLWriter(analysis_dir / "retrieval_chunks_detailed.jsonl")
            )
            long_writer = analysis_stack.enter_context(
                AtomicJSONLWriter(analysis_dir / "rerank_chunks.jsonl")
            )
            anomaly_writer = analysis_stack.enter_context(
                AtomicJSONLWriter(analysis_dir / "anomalies.jsonl")
            )

        with AtomicJSONLWriter(raw_path) as raw_writer, AtomicJSONLWriter(
            retrieval_path
        ) as retrieval_writer:

            def parse_function(path: Path) -> ParsedContext:
                try:
                    return parse_context_file(path, external_metadata)
                except Exception as exc:
                    return ParsedContext(
                        document=None,
                        units=[],
                        anomalies=[
                            {
                                "severity": "error",
                                "code": "context_processing_failure",
                                "source_file": str(path),
                                "message": f"{type(exc).__name__}: {exc}",
                            }
                        ],
                    )

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_workers
            ) as executor:
                parsed_iterator = bounded_thread_map(
                    executor,
                    parse_function,
                    context_files,
                    max_pending=config.max_workers * 3,
                )
                for file_index, parsed in enumerate(parsed_iterator, 1):
                    for raw_anomaly in parsed.anomalies:
                        anomaly = dict(raw_anomaly)
                        if anomaly.get("code") == "empty_passage":
                            empty_context_id = normalize_id(
                                anomaly.get("context_id")
                            )
                            if empty_context_id:
                                empty_context_ids.add(empty_context_id)


                            if (
                                empty_context_id not in relevant_context_ids
                                or config.empty_labeled_policy == "quarantine"
                            ):
                                anomaly["severity"] = "warning"
                        collect_anomaly(anomaly)
                        if anomaly_writer is not None:
                            anomaly_writer.write(anomaly)
                    document = parsed.document
                    if document is None:
                        counts["skipped_context_files"] += 1
                        continue
                    if document.context_id in seen_context_ids:
                        duplicate = {
                            "severity": "error",
                            "code": "duplicate_context_id",
                            "context_id": document.context_id,
                        }
                        collect_anomaly(duplicate)
                        if anomaly_writer is not None:
                            anomaly_writer.write(duplicate)
                        continue
                    seen_context_ids.add(document.context_id)
                    raw_writer.write(compact_raw_record(document))

                    long_chunks = (
                        build_long_chunks(
                            document, parsed.units, config, segmenter
                        )
                        if (
                            config.retrieval_profile == "quality"
                            or config.analysis_level == "full"
                        )
                        else []
                    )
                    packed_chunks = build_packed_retrieval_chunks(
                        document,
                        parsed.units,
                        long_chunks,
                        config,
                        segmenter,
                    )
                    context_text_hashes: set[str] = set()
                    for packed_chunk in packed_chunks:
                        chunk_id = str(packed_chunk["chunk_id"])
                        if chunk_id in seen_chunk_ids:
                            duplicate = {
                                "severity": "error",
                                "code": "duplicate_chunk_id",
                                "chunk_id": chunk_id,
                            }
                            collect_anomaly(duplicate)
                            if anomaly_writer is not None:
                                anomaly_writer.write(duplicate)
                            continue
                        seen_chunk_ids.add(chunk_id)
                        light = compact_retrieval_record(
                            document.context_id,
                            chunk_id,
                            str(packed_chunk["dense_text"]),
                        )
                        if is_heading_only_retrieval_candidate(
                            light["chunk_id"], light["text"]
                        ):
                            counts["heading_only_candidates_suppressed"] += 1
                            continue
                        text_hash = sha256_text(light["text"])
                        if text_hash in context_text_hashes:
                            counts["duplicate_views_suppressed"] += 1
                            continue
                        context_text_hashes.add(text_hash)
                        retrieval_writer.write(light)
                        context_ids_with_retrieval.add(document.context_id)
                        counts["packed_chunks"] += 1

                        counts["short_chunks"] += 1
                        counts["retrieval_records"] += 1
                        packed_body_lengths.add(
                            int(packed_chunk["body_char_count"])
                        )
                        packed_text_lengths.add(len(light["text"]))
                        if detailed_writer is not None:
                            detailed_writer.write(packed_chunk)

                    if config.retrieval_profile == "quality":
                        for article_chunk in long_chunks:
                            chunk_id = str(article_chunk["chunk_id"])
                            if chunk_id in seen_chunk_ids:
                                duplicate = {
                                    "severity": "error",
                                    "code": "duplicate_chunk_id",
                                    "chunk_id": chunk_id,
                                }
                                collect_anomaly(duplicate)
                                if anomaly_writer is not None:
                                    anomaly_writer.write(duplicate)
                                continue
                            seen_chunk_ids.add(chunk_id)
                            light = compact_retrieval_record(
                                document.context_id,
                                chunk_id,
                                str(article_chunk["rerank_text"]),
                            )
                            if is_heading_only_retrieval_candidate(
                                light["chunk_id"], light["text"]
                            ):
                                counts["heading_only_candidates_suppressed"] += 1
                                continue
                            text_hash = sha256_text(light["text"])
                            if text_hash in context_text_hashes:
                                counts["duplicate_views_suppressed"] += 1
                                continue
                            context_text_hashes.add(text_hash)
                            retrieval_writer.write(light)
                            context_ids_with_retrieval.add(document.context_id)
                            counts["article_chunks"] += 1
                            counts["retrieval_records"] += 1
                            article_body_lengths.add(
                                int(article_chunk["body_char_count"])
                            )
                            article_text_lengths.add(len(light["text"]))


                    for light in administrative_recovery_records(
                        document,
                        config,
                    ):
                        chunk_id = light["chunk_id"]
                        if chunk_id in seen_chunk_ids:
                            duplicate = {
                                "severity": "error",
                                "code": "duplicate_chunk_id",
                                "chunk_id": chunk_id,
                            }
                            collect_anomaly(duplicate)
                            if anomaly_writer is not None:
                                anomaly_writer.write(duplicate)
                            continue
                        seen_chunk_ids.add(chunk_id)
                        text_hash = sha256_text(light["text"])
                        if text_hash in context_text_hashes:
                            counts["duplicate_views_suppressed"] += 1
                            continue
                        context_text_hashes.add(text_hash)
                        retrieval_writer.write(light)
                        context_ids_with_retrieval.add(document.context_id)
                        counts["recovery_chunks"] += 1
                        counts["retrieval_records"] += 1
                        recovery_body_lengths.add(
                            len(retrieval_content_body(light["text"]))
                        )
                        recovery_text_lengths.add(len(light["text"]))

                    metadata_text = retrieval_metadata_text(
                        document,
                        parsed.units,
                        segmenter,
                        config.metadata_max_chars,
                    )
                    if metadata_text:
                        metadata_id = f"ctx{document.context_id}__metadata"
                        if metadata_id not in seen_chunk_ids:
                            seen_chunk_ids.add(metadata_id)
                            light = compact_retrieval_record(
                                document.context_id,
                                metadata_id,
                                metadata_text,
                            )
                            text_hash = sha256_text(light["text"])
                            if text_hash in context_text_hashes:
                                counts["duplicate_views_suppressed"] += 1
                            else:
                                context_text_hashes.add(text_hash)
                                retrieval_writer.write(light)
                                context_ids_with_retrieval.add(document.context_id)
                                counts["metadata_records"] += 1
                                counts["retrieval_records"] += 1

                    if unit_writer is not None:
                        for unit in parsed.units:
                            unit_writer.write(unit_record(unit))
                    if long_writer is not None:
                        for chunk in long_chunks:
                            long_writer.write(chunk)

                    for unit in parsed.units:
                        counts["legal_units"] += 1
                        unit_kind_counts[unit.kind] += 1
                        if unit.indexable:
                            counts["indexable_units"] += 1
                        else:
                            counts["excluded_units"] += 1
                            excluded_reason_counts[
                                unit.excluded_reason or "unspecified"
                            ] += 1
                    counts["documents"] += 1
                    title_source_counts[document.title_source] += 1
                    if file_index % 1000 == 0:
                        LOGGER.info(
                            "Processed %d/%d contexts; %d compact records.",
                            file_index,
                            len(context_files),
                            counts["retrieval_records"],
                        )

    quarantined_empty_contexts = (
        relevant_context_ids & empty_context_ids
        if config.empty_labeled_policy == "quarantine"
        else set()
    )
    missing_answer_contexts = sorted(
        relevant_context_ids
        - seen_context_ids
        - quarantined_empty_contexts,
        key=numeric_sort_key,
    )
    for context_id in missing_answer_contexts:
        collect_anomaly(
            {
                "severity": "error",
                "code": "answer_context_not_found",
                "context_id": context_id,
            }
        )
    missing_retrieval_contexts = sorted(
        relevant_context_ids
        - context_ids_with_retrieval
        - quarantined_empty_contexts,
        key=numeric_sort_key,
    )
    for context_id in missing_retrieval_contexts:

        if context_id in missing_answer_contexts:
            continue
        collect_anomaly(
            {
                "severity": "error",
                "code": "answer_context_without_retrieval_candidate",
                "context_id": context_id,
            }
        )

    query_groups = _query_output_groups(queries)
    for dataset, dataset_queries in query_groups.items():
        write_jsonl(
            queries_dir / f"{dataset}.jsonl",
            (
                {"query_id": query.query_id, "question": query.question}
                for query in dataset_queries
            ),
        )
        qrel_rows = [
            {"query_id": query.query_id, "context_id": context_id}
            for query in dataset_queries
            for context_id in query.answer_context_ids
        ]
        write_jsonl(qrels_dir / f"{dataset}.jsonl", qrel_rows)
        counts["queries"] += len(dataset_queries)
        counts["qrels"] += len(qrel_rows)

    files = {
        "raw_corpus": str(raw_path),
        "retrieval_corpus": str(retrieval_path),
        "queries": [str(queries_dir / f"{name}.jsonl") for name in query_groups],
        "qrels": [str(qrels_dir / f"{name}.jsonl") for name in query_groups],
    }
    file_sizes = {
        key: (
            [Path(path).stat().st_size for path in value]
            if isinstance(value, list)
            else Path(value).stat().st_size
        )
        for key, value in files.items()
    }
    manifest = {
        "pipeline": "Vietnamese Legal IR compact preprocessing",
        "pipeline_version": PIPELINE_VERSION,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "output_mode": "compact",
        "config": _serialized_build_config(config),
        "counts": dict(counts),
        "quality": {
            "anomaly_counts": dict(anomaly_counts),
            "anomaly_examples": anomaly_examples,
            "error_examples": error_examples,
            "excluded_reasons": dict(excluded_reason_counts),
            "unit_kinds": dict(unit_kind_counts),
            "title_sources": dict(title_source_counts),
            "missing_answer_contexts": missing_answer_contexts,
            "missing_retrieval_contexts": missing_retrieval_contexts,
            "quarantined_empty_contexts": sorted(
                quarantined_empty_contexts,
                key=numeric_sort_key,
            ),
            "packed_body_chars": packed_body_lengths.summary(),
            "packed_text_chars": packed_text_lengths.summary(),
            "article_body_chars": article_body_lengths.summary(),
            "article_text_chars": article_text_lengths.summary(),
            "recovery_body_chars": recovery_body_lengths.summary(),
            "recovery_text_chars": recovery_text_lengths.summary(),
            "packed_body_violations": sum(
                count
                for length, count in packed_body_lengths.frequencies.items()
                if length > config.retrieval_body_chars
            ),
            "article_body_violations": sum(
                count
                for length, count in article_body_lengths.frequencies.items()
                if length > config.long_body_chars
            ),
            "recovery_body_violations": sum(
                count
                for length, count in recovery_body_lengths.frequencies.items()
                if length > config.retrieval_body_chars
            ),
            "lossless_recovery": {
                "primary_view_hidden_chars": counts.get(
                    "administrative_removed_chars",
                    0,
                ),
                "source_chars_routed_to_recovery": counts.get(
                    "administrative_recovery_source_chars",
                    0,
                ),
                "normalization_only_char_delta": max(
                    0,
                    counts.get("administrative_removed_chars", 0)
                    - counts.get("administrative_recovery_source_chars", 0),
                ),
                "recovery_chunks": counts.get("recovery_chunks", 0),
                "interpretation": (
                    "Administrative spans are hidden only from clean primary "
                    "views and remain indexable through lossless_recovery."
                ),
            },
        },
        "files": files,
        "file_sizes_bytes": file_sizes,
        "retrieval_schema": ["chunk_id", "context_id", "text"],
        "changes_applied": [
            "Removed only high-confidence Nơi nhận/signature spans.",
            "Preserved uncertain or disproportionately large administrative tails.",
            "Recovered attached titles flattened after signatures, including CHƯƠNG TRÌNH, NỘI QUY and PHỤ LỤC.",
            "Indexed every removed administrative span in compact lossless-recovery chunks so cleanup cannot hide source content.",
            "Propagated short clause headings into descendant hierarchy.",
            "Kept headings in prefixes and suppressed standalone heading-only candidates.",
            "Packed adjacent legal atoms within one article to reduce tiny chunks.",
            "Added a complementary article-window view in quality profile.",
            "Added a factual law-level overview from metadata and headings.",
            "Used guarded inline restart detection without matching recipient rows such as Bộ Kế hoạch or Như Điều 3.",
            "Validated that every labeled context has at least one retrieval candidate.",
            "Downgraded irrelevant empty source files to explicit warnings.",
            "Quarantined labeled empty source files without deleting qrels or aborting a completed corpus build.",
            "Stored source passage once and retrieval text once per candidate.",
            "Folded document metadata into the same compact retrieval corpus.",
        ],
        "usage": {
            "index_file": str(retrieval_path),
            "document_key": "context_id",
            "candidate_key": "chunk_id",
            "text_field": "text",
            "aggregate_by": "context_id",
            "candidate_views": (
                ["packed", "article_window", "lossless_recovery", "metadata"]
                if config.retrieval_profile == "quality"
                else ["packed", "lossless_recovery", "metadata"]
            ),
            "recommended_chunk_aggregation": "max_per_context_then_rank",
        },
    }
    write_json(config.output_dir / "manifest.json", manifest)
    validation = validate_compact_workspace(config.output_dir)
    manifest = load_json(config.output_dir / "manifest.json")
    LOGGER.info(
        "Compact build complete: %d documents, %d retrieval records, %.2f GiB.",
        counts["documents"],
        counts["retrieval_records"],
        retrieval_path.stat().st_size / (1024**3),
    )
    if config.strict and (
        anomaly_counts["error"] or validation["status"] != "passed"
    ):
        error_codes = sorted(
            key.removeprefix("code::")
            for key, value in anomaly_counts.items()
            if key.startswith("code::") and value
            and any(
                item.get("code") == key.removeprefix("code::")
                for item in error_examples
            )
        )
        raise RuntimeError(
            "Strict validation failed: "
            f"data_errors={anomaly_counts['error']}, "
            f"validation_errors={len(validation['errors'])}, "
            f"codes={error_codes or ['referential_integrity']}, "
            f"missing_answer_contexts={missing_answer_contexts}; "
            f"missing_retrieval_contexts={missing_retrieval_contexts}; "
            "see manifest.json"
        )
    return manifest


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSONL at {path}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"JSONL record must be an object at {path}:{line_number}"
                )
            yield value


def validate_compact_workspace(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / "manifest.json"
    manifest = load_json(manifest_path)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    quarantined_empty_contexts = {
        normalize_id(value)
        for value in manifest.get("quality", {}).get(
            "quarantined_empty_contexts",
            [],
        )
        if normalize_id(value)
    }

    def issue(code: str, **details: Any) -> None:
        if len(errors) < 100:
            errors.append({"code": code, **details})

    def warn(code: str, **details: Any) -> None:
        if len(warnings) < 100:
            warnings.append({"code": code, **details})

    raw_path = workspace / "corpus" / "corpus_raw.jsonl"
    retrieval_path = workspace / "corpus" / "retrieval_chunks.jsonl"
    for path in (raw_path, retrieval_path):
        if not path.is_file():
            issue("missing_required_file", path=str(path))
    if errors:
        return {"status": "failed", "errors": errors, "warnings": warnings}

    context_ids: set[str] = set()
    raw_count = 0
    for row in iter_jsonl(raw_path):
        raw_count += 1
        context_id = normalize_id(row.get("id", row.get("context_id")))
        if not context_id:
            issue("raw_missing_context_id", row=raw_count)
        elif context_id in context_ids:
            issue("duplicate_context_id", context_id=context_id)
        else:
            context_ids.add(context_id)
        if not normalize_inline(row.get("passage", "")):
            issue("raw_empty_passage", context_id=context_id)

    chunk_ids: set[str] = set()
    contexts_with_chunks: set[str] = set()
    candidate_view_counts: Counter[str] = Counter()
    retrieval_count = 0
    exact_schema = {"chunk_id", "context_id", "text"}
    for row in iter_jsonl(retrieval_path):
        retrieval_count += 1
        if set(row) != exact_schema:
            issue(
                "non_compact_retrieval_schema",
                row=retrieval_count,
                fields=sorted(row),
            )
        chunk_id = normalize_id(row.get("chunk_id"))
        context_id = normalize_id(row.get("context_id"))
        if not chunk_id or chunk_id in chunk_ids:
            issue("duplicate_or_empty_chunk_id", chunk_id=chunk_id)
        else:
            chunk_ids.add(chunk_id)
        if context_id not in context_ids:
            issue(
                "retrieval_unknown_context",
                chunk_id=chunk_id,
                context_id=context_id,
            )
        else:
            contexts_with_chunks.add(context_id)
        if not normalize_inline(row.get("text", "")):
            issue("retrieval_empty_text", chunk_id=chunk_id)
        view = compact_candidate_view(chunk_id)
        candidate_view_counts[view] += 1
        if is_heading_only_retrieval_candidate(chunk_id, row.get("text", "")):
            issue(
                "heading_only_retrieval_candidate",
                chunk_id=chunk_id,
                context_id=context_id,
            )

    query_keys: set[tuple[str, str]] = set()
    query_count = 0
    for path in sorted((workspace / "queries").glob("*.jsonl")):
        for row in iter_jsonl(path):
            query_count += 1
            if set(row) != {"query_id", "question"}:
                issue(
                    "non_compact_query_schema",
                    path=str(path),
                    fields=sorted(row),
                )
            query_id = normalize_id(row.get("query_id"))
            if not query_id or not normalize_inline(row.get("question", "")):
                issue("invalid_query", path=str(path), query_id=query_id)
            query_keys.add((path.stem, query_id))

    qrel_count = 0
    for path in sorted((workspace / "qrels").glob("*.jsonl")):
        for row in iter_jsonl(path):
            qrel_count += 1
            if set(row) != {"query_id", "context_id"}:
                issue(
                    "non_compact_qrel_schema",
                    path=str(path),
                    fields=sorted(row),
                )
            query_id = normalize_id(row.get("query_id"))
            context_id = normalize_id(row.get("context_id"))
            if (path.stem, query_id) not in query_keys:
                issue("qrel_unknown_query", query_id=query_id, dataset=path.stem)
            if context_id not in context_ids:
                if context_id in quarantined_empty_contexts:
                    warn(
                        "qrel_quarantined_empty_context",
                        query_id=query_id,
                        context_id=context_id,
                    )
                else:
                    issue(
                        "qrel_unknown_context",
                        query_id=query_id,
                        context_id=context_id,
                    )
            elif context_id not in contexts_with_chunks:
                issue(
                    "qrel_context_without_retrieval_candidate",
                    query_id=query_id,
                    context_id=context_id,
                )

    expected = manifest.get("counts", {})
    actual = {
        "documents": raw_count,
        "retrieval_records": retrieval_count,
        "queries": query_count,
        "qrels": qrel_count,
    }
    for key, value in actual.items():
        if key in expected and int(expected[key]) != value:
            issue(
                "manifest_count_mismatch",
                field=key,
                expected=int(expected[key]),
                actual=value,
            )
    report = {
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed" if not errors else "failed",
        "counts": actual,
        "candidate_views": dict(candidate_view_counts),
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "retrieval_schema_exactly_three_fields": not any(
                item["code"] == "non_compact_retrieval_schema" for item in errors
            ),
            "context_ids_unique": raw_count == len(context_ids),
            "chunk_ids_unique": retrieval_count == len(chunk_ids),
            "heading_only_candidates_absent": not any(
                item["code"] == "heading_only_retrieval_candidate"
                for item in errors
            ),
            "all_qrel_contexts_retrievable": not any(
                item["code"] in {
                    "qrel_unknown_context",
                    "qrel_context_without_retrieval_candidate",
                    "qrel_quarantined_empty_context",
                }
                for item in [*errors, *warnings]
            ),
            "all_non_quarantined_qrel_contexts_retrievable": not any(
                item["code"] in {
                    "qrel_unknown_context",
                    "qrel_context_without_retrieval_candidate",
                }
                for item in errors
            ),
            "query_label_leakage_absent": not any(
                item["code"] == "non_compact_query_schema" for item in errors
            ),
        },
    }
    manifest["validation"] = report
    write_json(manifest_path, manifest)
    return report


def validate_workspace(workspace: Path) -> dict[str, Any]:
    manifest_path = workspace / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    manifest = load_json(manifest_path)
    if manifest.get("output_mode") == "compact":
        return validate_compact_workspace(workspace)
    raise ValueError(
        "Unsupported workspace: manifest.json has no output_mode=compact. "
        "Only compact workspaces produced by this pipeline are supported."
    )


def self_test() -> None:
    legal_document = {
        "id": 21,
        "link": (
            "https://example.org/Quyet-dinh-36-2012-QD-TTg-"
            "Quy-che-chat-luong-147304.aspx"
        ),
        "name": (
            "Quyet-dinh-36-2012-QD-TTg-Quy-che-chat-luong-147304"
        ),
        "passage": (
            "THỦ TƯỚNG CHÍNH\n\nPHỦ\n\n"
            "Số: 36/2012/QĐ-TTg\n\n"
            "QUYẾT ĐỊNH\n\n"
            "BAN HÀNH QUY CHẾ VỀ CHẤT LƯỢNG CÔNG TRÌNH\n\n"
            "Căn cứ Luật tổ chức Chính phủ;\n\n"
            "Chương I\n\nQUY ĐỊNH CHUNG\n\n"
            "Điều 1. Phạm vi điều chỉnh\n\n"
            "1. Quy chế này quy định việc đánh giá chất lượng.\n\n"
            "2. Đối tượng áp dụng bao gồm:\n\n"
            "a) Cơ quan nhà nước;\n\n"
            "b) Tổ chức có liên quan.\n\n"
            "Điều 2. Trách nhiệm thực hiện\n\n"
            "1. Việc thực hiện tuân theo khoản 1 Điều 1.\n\n"
            "2. Cơ quan có trách nhiệm báo cáo kết quả.\n\n"
            "Nơi nhận:\n\n- Văn phòng Chính phủ;\n\n- Lưu: VT."
        ),
    }
    standard_document = {
        "id": 69,
        "link": (
            "https://example.org/TCVN-ISO-IEC-17020-2012-"
            "Danh-gia-su-phu-hop-914763.aspx"
        ),
        "passage": (
            "TIÊU CHUẨN QUỐC\n\nGIA\n\n"
            "TCVN ISO/IEC 17020:2012\n\n"
            "ĐÁNH GIÁ SỰ PHÙ HỢP - YÊU CẦU ĐỐI VỚI TỔ CHỨC "
            "GIÁM ĐỊNH\n\n"
            "Lời nói đầu\n\n"
            "Tiêu chuẩn này quy định các yêu cầu chung.\n\n"
            "1. Phạm vi áp dụng\n\n"
            "Tiêu chuẩn này áp dụng cho các tổ chức giám định.\n\n"
            "2. Tài liệu viện dẫn\n\n"
            "Áp dụng TCVN ISO/IEC 17000.\n\n"
            "3.1. Giám định\n\n"
            "Việc kiểm tra sản phẩm và xác định sự phù hợp.\n\n"
            "4.1. Tính khách quan\n\n"
            "Hoạt động giám định phải được thực hiện khách quan."
        ),
    }
    queries = {
        "100": {
            "question": (
                "Quy chế đánh giá chất lượng áp dụng cho đối tượng nào?"
            ),
            "answer": ["21"],
        },
        "101": {
            "question": "Tổ chức giám định phải bảo đảm điều gì?",
            "answer": ["69"],
        },
    }

    with tempfile.TemporaryDirectory(prefix="legalir-self-test-") as temp:
        root = Path(temp)
        contexts = root / "selected-contexts"
        contexts.mkdir()
        write_json(contexts / "context_21.json", legal_document)
        write_json(contexts / "context_69.json", standard_document)
        query_path = root / "train.json"
        write_json(query_path, queries)
        output = root / "workspace"

        compact_output = root / "workspace_compact"


        write_json(
            contexts / "context_999.json",
            {"id": 999, "passage": "", "name": "empty placeholder"},
        )
        compact_manifest = run_compact_build(
            BuildConfig(
                contexts_dir=contexts,
                query_files=[query_path],
                labeled_query_files=[query_path],
                output_dir=compact_output,
                short_body_chars=180,
                retrieval_body_chars=360,
                retrieval_overlap_chars=40,
                long_body_chars=500,
                max_workers=2,
                strict=True,
            )
        )
        compact_validation = validate_compact_workspace(compact_output)
        assert compact_validation["status"] == "passed", compact_validation
        compact_rows = list(
            iter_jsonl(compact_output / "corpus" / "retrieval_chunks.jsonl")
        )
        assert compact_manifest["counts"]["documents"] == 2
        assert compact_manifest["counts"]["packed_chunks"] >= 2
        assert compact_manifest["counts"]["article_chunks"] >= 2
        assert compact_manifest["counts"]["recovery_chunks"] >= 1
        assert compact_manifest["quality"]["lossless_recovery"][
            "source_chars_routed_to_recovery"
        ] > 0
        assert compact_manifest["counts"]["skipped_context_files"] == 1
        assert (
            compact_manifest["quality"]["anomaly_counts"]["warning"] >= 1
        )
        assert compact_rows
        assert all(set(row) == {"chunk_id", "context_id", "text"} for row in compact_rows)
        recovery_rows = [
            row for row in compact_rows if "__r" in row["chunk_id"]
        ]
        assert recovery_rows
        assert any(
            "Văn phòng Chính phủ" in row["text"]
            for row in recovery_rows
        )

        empty_query_path = root / "empty-with-answers.json"
        write_json(
            empty_query_path,
            {
                "800": {
                    "question": "Nguồn rỗng này quy định nội dung gì?",
                    "answer": ["999"],
                }
            },
        )
        empty_output = root / "workspace_empty_quarantine"
        empty_manifest = run_compact_build(
            BuildConfig(
                contexts_dir=contexts,
                query_files=[empty_query_path],
                labeled_query_files=[empty_query_path],
                output_dir=empty_output,
                short_body_chars=180,
                retrieval_body_chars=360,
                retrieval_overlap_chars=40,
                long_body_chars=500,
                max_workers=2,
                strict=True,
                empty_labeled_policy="quarantine",
            )
        )
        assert empty_manifest["quality"][
            "quarantined_empty_contexts"
        ] == ["999"]
        empty_validation = validate_compact_workspace(empty_output)
        assert empty_validation["status"] == "passed", empty_validation
        assert empty_validation["warnings"] == [
            {
                "code": "qrel_quarantined_empty_context",
                "query_id": "800",
                "context_id": "999",
            }
        ]
        assert not empty_validation["checks"][
            "all_qrel_contexts_retrievable"
        ]
        assert empty_validation["checks"][
            "all_non_quarantined_qrel_contexts_retrievable"
        ]

        missing_query_path = root / "missing-with-answers.json"
        write_json(
            missing_query_path,
            {
                "900": {
                    "question": "Văn bản bị thiếu quy định điều gì?",
                    "answer": ["123456789"],
                }
            },
        )
        missing_output = root / "workspace_missing"
        try:
            run_compact_build(
                BuildConfig(
                    contexts_dir=contexts,
                    query_files=[missing_query_path],
                    labeled_query_files=[missing_query_path],
                    output_dir=missing_output,
                    short_body_chars=180,
                    retrieval_body_chars=360,
                    retrieval_overlap_chars=40,
                    long_body_chars=500,
                    max_workers=2,
                    strict=True,
                )
            )
        except RuntimeError as exc:
            assert "123456789" in str(exc)
        else:
            raise AssertionError("Strict mode accepted a missing qrel context")
        missing_manifest = load_json(missing_output / "manifest.json")
        assert missing_manifest["validation"]["status"] == "failed"
        assert missing_manifest["quality"]["missing_answer_contexts"] == [
            "123456789"
        ]
        assert not missing_manifest["validation"]["checks"][
            "all_qrel_contexts_retrievable"
        ]

        cleaned, admin_stats = remove_administrative_blocks(
            [
                "Điều 2. Có hiệu lực.\nNơi nhận:",
                "- Văn phòng Quốc hội;",
                "- Lưu: VT.\nTHỦ TƯỚNG\nQUY CHẾ\nTỔ CHỨC XÉT THƯỞNG",
                "Điều 1. Phạm vi điều chỉnh",
            ]
        )
        assert admin_stats["removed_segments"] >= 2
        assert not any("Văn phòng Quốc hội" in item for item in cleaned)
        assert any(item.startswith("QUY CHẾ") for item in cleaned)


        fused_program, fused_program_stats = remove_administrative_blocks(
            [
                "Điều 3. Các cơ quan chịu trách nhiệm thi hành.",
                "Nơi nhận:",
                "- Bộ Kế hoạch và Đầu tư;",
                "- Lưu: VT.",
                "TM. CHÍNH PHỦ KT. THỦ TƯỚNG PHÓ THỦ TƯỚNG "
                "Nguyễn Văn A CHƯƠNG TRÌNH HÀNH ĐỘNG CỦA CHÍNH PHỦ "
                "(Kèm theo Nghị quyết số 01/NQ-CP)",
                "I. MỤC ĐÍCH, YÊU CẦU",
                "1. Tổ chức thực hiện đầy đủ các nhiệm vụ.",
            ]
        )
        assert not any("Nơi nhận" in item for item in fused_program)
        assert any(
            item.startswith("CHƯƠNG TRÌNH HÀNH ĐỘNG")
            for item in fused_program
        )
        assert any("I. MỤC ĐÍCH" in item for item in fused_program)
        assert fused_program_stats["inline_restart_count"] == 1


        fused_rules, fused_rules_stats = remove_administrative_blocks(
            [
                "Điều 3. Quyết định có hiệu lực kể từ ngày ký.",
                "Nơi nhận:",
                "- Như Điều 3;",
                "- Lưu: VT.",
                "TỔNG CỤC TRƯỞNG Cao Anh Tuấn NỘI QUY TIẾP CÔNG DÂN "
                "(Kèm theo Quyết định số 1607/QĐ-TCT)",
                "I. NHỮNG QUY ĐỊNH CHUNG",
                "1. Công dân có trách nhiệm tuân thủ nội quy.",
            ]
        )
        assert any(item.startswith("NỘI QUY") for item in fused_rules)
        assert any("NHỮNG QUY ĐỊNH CHUNG" in item for item in fused_rules)
        assert fused_rules_stats["inline_restart_count"] == 1


        fused_appendix, appendix_stats = remove_administrative_blocks(
            [
                "Điều 15. Thông tư có hiệu lực thi hành.",
                "Nơi nhận:",
                "- Các bộ, ngành;",
                "- Lưu: VT.",
                "KT. BỘ TRƯỞNG THỨ TRƯỞNG Bùi Hồng Lĩnh PHỤ LỤC I "
                "DANH MỤC NHÀ TÙ (Kèm theo Thông tư số 16/2014/TT-BLĐTBXH)",
                "1. An Giang - Khám Long Xuyên.",
            ]
        )
        assert any(item.startswith("PHỤ LỤC I") for item in fused_appendix)
        assert any("Khám Long Xuyên" in item for item in fused_appendix)
        assert appendix_stats["inline_restart_count"] == 1


        false_restart, _ = find_administrative_restart(
            "- Bộ Kế hoạch và Đầu tư;"
        )
        assert false_restart is None


        ambiguous, ambiguous_stats = remove_administrative_blocks(
            [
                "Điều 2. Văn bản có hiệu lực.",
                "Nơi nhận:",
                "- Lưu: VT.",
                "TÀI LIỆU ĐÍNH KÈM ĐẶC BIỆT",
                "I. MỤC ĐÍCH, YÊU CẦU",
                "1. Nội dung pháp lý cần được bảo toàn.",
            ]
        )
        assert any("Nơi nhận" in item for item in ambiguous)
        assert any("Nội dung pháp lý" in item for item in ambiguous)
        assert ambiguous_stats["ambiguous_blocks_preserved"] == 1

    print("Self-test passed.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a validated retrieval-first Vietnamese legal corpus for "
            "BM25, BGE-M3 and context-level evaluation."
        )
    )
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="Build the compact five-file LegalIR workspace.",
    )
    build.add_argument(
        "--contexts-dir",
        type=Path,
        required=True,
        help="Directory containing context_*.json files.",
    )
    build.add_argument(
        "--queries",
        type=Path,
        nargs="*",
        default=[],
        help=(
            "One or more question files, e.g. train.json queries.json. "
            "May be omitted for corpus-only preprocessing."
        ),
    )
    build.add_argument(
        "--labeled-queries",
        type=Path,
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Subset of --queries whose 'answer' field may be read. Files not listed "
            "here are treated as questions only: their answers are ignored and no "
            "qrels rows are written for them. Leave empty when ranking an unlabeled "
            "set such as queries.json."
        ),
    )
    build.add_argument(
        "--output",
        type=Path,
        default=Path("workspace"),
        help="Output workspace directory (default: ./workspace).",
    )
    build.add_argument(
        "--metadata-jsonl",
        type=Path,
        help=(
            "Optional validated official/generated metadata JSON or JSONL "
            "keyed by context_id."
        ),
    )
    build.add_argument(
        "--retrieval-body-chars",
        type=int,
        default=1000,
        help=(
            "Maximum body characters for packed BM25/BGE-M3 passages."
        ),
    )
    build.add_argument(
        "--retrieval-overlap-chars",
        type=int,
        default=120,
        help="Overlap only when one legal unit exceeds retrieval-body-chars.",
    )
    build.add_argument("--long-body-chars", type=int, default=2000)
    build.add_argument("--long-overlap-units", type=int, default=1)
    build.add_argument("--long-overlap-chars", type=int, default=200)
    build.add_argument("--max-prefix-chars", type=int, default=500)
    build.add_argument(
        "--retrieval-profile",
        choices=("compact", "quality"),
        default="quality",
        help=(
            "quality writes packed + article-window + metadata views; "
            "compact omits article windows to minimize index size."
        ),
    )
    build.add_argument(
        "--metadata-max-chars",
        type=int,
        default=1800,
        help="Maximum characters in the single document-overview record.",
    )
    build.add_argument(
        "--max-workers",
        type=int,
        default=max(1, min(16, (os.cpu_count() or 4))),
    )
    build.add_argument(
        "--word-segmenter",
        choices=("simple", "underthesea", "vncorenlp"),
        default="simple",
    )
    build.add_argument("--vncorenlp-dir", type=Path)
    build.add_argument(
        "--allow-partial-json",
        action="store_true",
        help=(
            "Recover complete records from a truncated pasted query JSON. "
            "Do not use for the original valid dataset unless necessary."
        ),
    )
    build.add_argument(
        "--empty-labeled-policy",
        choices=("quarantine", "error"),
        default="quarantine",
        help=(
            "quarantine keeps qrels and completes the build with explicit "
            "validation warnings when an official labeled context has an "
            "empty passage; error fails fast. Use error for the final corpus "
            "after repairing all source files."
        ),
    )
    build.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero after writing reports if any data error is found.",
    )
    build.add_argument(
        "--analysis-level",
        choices=("none", "full"),
        default="none",
        help=(
            "none writes only the compact operational files; full also "
            "writes verbose legal-unit/chunk files under analysis/."
        ),
    )
    validate = subparsers.add_parser(
        "validate",
        help="Validate a completed workspace and detect label leakage.",
    )
    validate.add_argument(
        "--workspace",
        type=Path,
        required=True,
        help="Workspace previously produced by the build command.",
    )
    subparsers.add_parser(
        "self-test",
        help="Run deterministic end-to-end tests without external packages.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    if args.command == "self-test":
        self_test()
        return
    if args.command == "validate":
        report = validate_workspace(args.workspace)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if report["status"] != "passed":
            raise SystemExit(3)
        return
    if args.command == "build":
        config = BuildConfig(
            contexts_dir=args.contexts_dir,
            query_files=list(args.queries),
            labeled_query_files=list(args.labeled_queries),
            output_dir=args.output,
            retrieval_body_chars=args.retrieval_body_chars,
            retrieval_overlap_chars=args.retrieval_overlap_chars,
            long_body_chars=args.long_body_chars,
            long_overlap_units=args.long_overlap_units,
            long_overlap_chars=args.long_overlap_chars,
            max_prefix_chars=args.max_prefix_chars,
            max_workers=args.max_workers,
            word_segmenter=args.word_segmenter,
            vncorenlp_dir=args.vncorenlp_dir,
            metadata_jsonl=args.metadata_jsonl,
            allow_partial_json=args.allow_partial_json,
            strict=args.strict,
            analysis_level=args.analysis_level,
            retrieval_profile=args.retrieval_profile,
            metadata_max_chars=args.metadata_max_chars,
            empty_labeled_policy=args.empty_labeled_policy,
        )
        manifest = run_compact_build(config)
        print(
            json.dumps(
                manifest["counts"],
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
