"""Tinh chỉnh adapter LoRA cho F2LLM.

File này làm gì
----------------
Lấy bộ triplet đã chuẩn bị, chạy một epoch fine-tune LoRA trên F2LLM, lưu adapter đã
tinh chỉnh. Cần môi trường riêng vì adapter LoRA của F2LLM chỉ nạp được với
`sentence-transformers` 6.x.

Pipeline stage
--------------
Ngoài đường suy luận. Chỉ chạy khi muốn dựng lại checkpoint từ đầu.

Input
-----
  --train-data  file triplet
  --out-dir     thư mục ghi adapter
  --lr          learning rate
  --rank        hạng của LoRA
  --epoch       số epoch

Output
-------
  <out-dir>/   adapter LoRA đã tinh chỉnh

Model/class
-----------
  unsloth + sentence-transformers 6.x — `codefuse-ai/F2LLM-1.7B`. Cần CUDA.

Command
-------
  python run_f2llm_epoch.py --train-data <data> --out-dir checkpoints/f2llm_finetune_epoch2 \
      --lr <lr> --rank <r> --epoch 2
"""

from __future__ import annotations

import argparse
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--triplets-jsonl", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--prev-checkpoint", default="")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    from datasets import Dataset
    from unsloth import FastSentenceTransformer, is_bf16_supported
    from sentence_transformers import SentenceTransformer
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.losses import CachedGISTEmbedLoss
    from sentence_transformers.training_args import BatchSamplers

    rows = []
    with open(args.triplets_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    dataset = Dataset.from_list(rows)
    print(f"[run_f2llm_epoch] loaded {len(dataset)} triplet rows", flush=True)

    if args.epoch == 1:
        model = FastSentenceTransformer.from_pretrained(
            model_name="codefuse-ai/F2LLM-1.7B",
            max_seq_length=512,
            full_finetuning=False,
        )
        model = FastSentenceTransformer.get_peft_model(
            model,
            r=args.rank,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_alpha=64,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,
            random_state=3407,
            task_type="FEATURE_EXTRACTION",
        )
    else:
        if not args.prev_checkpoint:
            raise ValueError(f"epoch={args.epoch} requires --prev-checkpoint (resume)")
        model = FastSentenceTransformer.from_pretrained(
            args.prev_checkpoint,
            max_seq_length=512,
        )


    guide_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    guide_model.max_seq_length = 512


    _orig_guide_preprocess = guide_model.preprocess
    def _guide_preprocess_tensors_only(*a, **kw):
        out = _orig_guide_preprocess(*a, **kw)
        return {k: v for k, v in out.items() if hasattr(v, "to")}
    guide_model.preprocess = _guide_preprocess_tensors_only
    loss = CachedGISTEmbedLoss(model, guide=guide_model)

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=dataset,
        loss=loss,
        args=SentenceTransformerTrainingArguments(
            output_dir=args.out_dir,
            num_train_epochs=1,
            per_device_train_batch_size=128,
            gradient_accumulation_steps=1,
            learning_rate=args.lr,
            fp16=not is_bf16_supported(),
            bf16=is_bf16_supported(),
            logging_steps=50,
            warmup_ratio=0.03,
            report_to="none",
            lr_scheduler_type="constant_with_warmup",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
            dataloader_num_workers=6,
        ),
    )
    trainer.train()
    model.save_pretrained(args.out_dir)
    model.tokenizer.save_pretrained(args.out_dir)
    print(f"[run_f2llm_epoch] DONE epoch={args.epoch} saved to {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
