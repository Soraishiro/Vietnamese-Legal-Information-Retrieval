import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from legal_ir.preprocessing.preprocess import *
from legal_ir.preprocessing.preprocess import main

if __name__ == "__main__":
    main()