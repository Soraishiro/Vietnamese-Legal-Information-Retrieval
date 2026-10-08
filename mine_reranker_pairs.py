import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.training.mine_reranker_pairs import *
from legal_ir.training.mine_reranker_pairs import main

if __name__ == "__main__":
    raise SystemExit(main())