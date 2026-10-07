"""Fine-tune BGE-M3 một epoch trên dữ liệu flagembedding đã mine.

Metadata thay đổi
-----------------
2026-09-30

File này làm gì
--------------
Gọi bộ fine-tune có sẵn của thư viện FlagEmbedding
(`FlagEmbedding.finetune.embedder.encoder_only.m3`). BGE-M3 full fine-tune cần
unified finetuning + self-distillation, cả hai đã có trong thư viện.

Pipeline stage
--------------
Train embedder (bước ngoài pipeline suy luận, chỉ dùng khi muốn dựng lại
checkpoint). Dữ liệu đầu vào do `mine_dense_negatives.py` và
`train_dense_embedder.py filter` sinh ra.

Input
-----
- `--train-json`: file dạng `{query, pos: [...], neg: [...]}`, lấy từ
  `dense_evidence_train_v1.flagembedding.jsonl`.
- `--prev-checkpoint`: bỏ trống khi train epoch 1; epoch ≥ 2 nạp checkpoint
  epoch trước để nối tiếp.

Output
------
- `--out-dir`: checkpoint của epoch vừa train.

Model/class khởi tạo
--------------------
`BAAI/bge-m3` (hoặc `--model-base` khác) nạp bên trong tiến trình con do
FlagEmbedding quản lý.

Chạy
----
    python train_bge_embedder.py --train-json <...>.flagembedding.jsonl \
        --lr 2e-5 --out-dir <checkpoint_epoch1>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_MODEL_BASE = "BAAI/bge-m3"


def build_command(
    train_json: Path,
    out_dir: Path,
    learning_rate: float,
    model_path: str,
    extra: list[str],
) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "1",
        "-m",
        "FlagEmbedding.finetune.embedder.encoder_only.m3",
        "--model_name_or_path",
        model_path,
        "--train_data",
        str(train_json),
        "--train_group_size",
        "2",
        "--query_max_len",
        "512",
        "--passage_max_len",
        "1024",
        "--same_dataset_within_batch",
        "True",
        "--output_dir",
        str(out_dir),
        "--overwrite_output_dir",
        "--learning_rate",
        str(learning_rate),
        "--bf16",
        "True",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--gradient_checkpointing",
        "True",
        "--negatives_cross_device",
        "--temperature",
        "0.02",
        "--normalize_embeddings",
        "True",
        "--unified_finetuning",
        "True",
        "--use_self_distill",
        "True",
        "--self_distill_start_step",
        "0",
        "--gradient_accumulation_steps",
        "8",
        "--report_to",
        "none",
        *extra,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--model-base", default=DEFAULT_MODEL_BASE)
    parser.add_argument("--prev-checkpoint", default="")
    parser.add_argument(
        "--deepspeed-config",
        type=Path,
        help="Truyền --deepspeed <file>; bỏ trống thì chạy không DeepSpeed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In lệnh rồi thoát, không train.",
    )
    args = parser.parse_args()

    if not args.train_json.is_file():
        raise FileNotFoundError(args.train_json)
    model_path = args.prev_checkpoint or args.model_base
    extra: list[str] = []
    if args.deepspeed_config is not None:
        if not args.deepspeed_config.is_file():
            raise FileNotFoundError(args.deepspeed_config)
        extra += ["--deepspeed", str(args.deepspeed_config)]

    command = build_command(
        args.train_json, args.out_dir, args.lr, model_path, extra
    )
    print(" ".join(command), flush=True)
    if args.dry_run:
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
