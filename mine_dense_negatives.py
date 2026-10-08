import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.training.mine_dense_negatives import *
from legal_ir.training.mine_dense_negatives import main

if __name__ == "__main__":
    raise SystemExit(main())