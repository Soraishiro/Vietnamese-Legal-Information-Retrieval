import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.training.train_cross_encoder import *
from legal_ir.training.train_cross_encoder import main

if __name__ == "__main__":
    main()