import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.training.train_bge_embedder import *
from legal_ir.training.train_bge_embedder import main

if __name__ == "__main__":
    raise SystemExit(main())