#!/usr/bin/env python3
"""
Metadata thay đổi

File này làm gì
    Sinh `checkpoints/SHA256SUMS` — băm toàn bộ file trong `checkpoints/` để người nhận
    gói kiểm tra file tải về còn nguyên trước khi chạy.

Pipeline stage
    Không thuộc stage nào.

Input/output
    Input : checkpoints/ (3 thư mục con)
    Output: checkpoints/SHA256SUMS, một dòng `<sha256>  ./<đường/dẫn tương đối>` mỗi file

Model/class
    Không có.

Command
    python make_checkpoints_sha256sums.py
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUT = CHECKPOINTS_DIR / "SHA256SUMS"


def sha256_file(path: Path) -> str:
    """SHA-256 theo byte, đọc streaming để không nạp file 2 GB vào RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    files = sorted(p for p in CHECKPOINTS_DIR.rglob("*") if p.is_file() and p != OUTPUT)
    lines = [
        f"{sha256_file(path)}  ./{path.relative_to(CHECKPOINTS_DIR).as_posix()}"
        for path in files
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(lines)} file -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
