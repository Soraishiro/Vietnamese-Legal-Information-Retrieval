import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.tools.make_checkpoints_sha256sums import *
from legal_ir.tools.make_checkpoints_sha256sums import main

if __name__ == "__main__":
    raise SystemExit(main())