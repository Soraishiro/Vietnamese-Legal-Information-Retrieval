import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.retrieval.apply_ranking_rules import *
from legal_ir.retrieval.apply_ranking_rules import main

if __name__ == "__main__":
    raise SystemExit(main())