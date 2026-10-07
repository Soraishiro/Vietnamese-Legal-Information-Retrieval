"""Huấn luyện cross-encoder để xếp lại ứng viên.

File này làm gì
----------------
Lấy các cặp (câu hỏi, ứng viên tích cực, ứng viên tiêu cực) đã khai thác sẵn, chia một
phần ra làm tập kiểm trong lúc huấn luyện, rồi huấn luyện theo từng epoch và lưu checkpoint
mỗi epoch.

Pipeline stage
--------------
Ngoài đường suy luận. Chỉ chạy khi muốn dựng lại checkpoint từ đầu.

Input
-----
  --data    file cặp train
  --out-dir thư mục ghi checkpoint
  Cùng các tham số: learning rate, số epoch, kích thước lô, seed, tỉ lệ tập kiểm

Output
-------
  <out-dir>/epoch<N>/   checkpoint của từng epoch, kèm manifest và cấu hình huấn luyện

Model/class
-----------
  transformers.AutoModelForSequenceClassification — khởi đầu từ `BAAI/bge-reranker-v2-m3`.
  Chỉ chạy trên CUDA.

Command
-------
  python train_cross_encoder.py --data <pairs.jsonl> --out-dir checkpoints/ce
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)


@dataclass
class Config:
    model: str = "BAAI/bge-reranker-v2-m3"
    data: str = "training_data/reranker_pairs/train_pairs_dieu.jsonl"
    out: str = "training/checkpoints/reranker"
    summary: str = "outputs/tables/train_cross_encoder.json"
    group_size: int = 8
    max_query_len: int = 96
    max_passage_len: int = 512


    lr: float = 6e-5
    epochs: int = 3
    batch_size: int = 4
    grad_accum: int = 4
    warmup_ratio: float = 0.05
    seed: int = 20260819
    val_fraction: float = 0.1


    @property
    def pairs_per_forward(self) -> int:
        return self.batch_size * self.group_size


class GroupDataset(Dataset):

    def __init__(self, rows: list[dict], group_size: int, rng: random.Random):
        self.rows = rows
        self.g = group_size
        self.rng = rng

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        negs = list(r["neg"])
        need = self.g - 1
        if len(negs) >= need:
            negs = self.rng.sample(negs, need)
        else:
            negs = negs + [self.rng.choice(negs) for _ in range(need - len(negs))]
        return r["query"], [r["pos"][0]] + negs


def make_collate(tok, cfg: Config):
    def collate(batch):
        queries, passages = [], []
        for q, group in batch:
            for p in group:
                queries.append(q)
                passages.append(p)
        enc = tok(queries, passages, truncation="only_second", padding=True,
                  max_length=cfg.max_query_len + cfg.max_passage_len,
                  return_tensors="pt")
        return enc, len(batch)
    return collate


def evaluate(model, loader, device, group_size) -> tuple[float, float]:
    model.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for enc, n in loader:
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.view(n, group_size)
            labels = torch.zeros(n, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits, labels).item())
            correct += (logits.argmax(dim=1) == 0).sum().item()
            total += n
    model.train()
    return float(np.mean(losses)), correct / max(total, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    for f, t in (("model", str), ("data", str), ("out", str), ("summary", str)):
        ap.add_argument(f"--{f}", type=t)
    for f, t in (("group_size", int), ("max_query_len", int), ("max_passage_len", int),
                 ("lr", float), ("epochs", int), ("batch_size", int),
                 ("grad_accum", int), ("warmup_ratio", float), ("seed", int)):
        ap.add_argument(f"--{f.replace('_', '-')}", type=t)
    args = ap.parse_args()
    cfg = Config(**{k.replace("-", "_"): v for k, v in vars(args).items() if v is not None})

    if cfg.pairs_per_forward > 64:
        raise SystemExit(f"pairs/forward={cfg.pairs_per_forward} > 64 — quá VRAM A100; "
                         f"hạ batch_size hoặc group_size (xem docstring)")
    if cfg.pairs_per_forward > 32:
        print(f"WARNING: pairs/forward={cfg.pairs_per_forward} > 32 — cần A100 (60 GB), "
              f"không vừa L40", flush=True)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = random.Random(cfg.seed)


    with Path(cfg.data).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    print(f"loaded {len(rows)} training rows from {cfg.data}", flush=True)

    rng.shuffle(rows)
    n_val = int(len(rows) * cfg.val_fraction)
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"train {len(train_rows)}  val {len(val_rows)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise SystemExit("Không có CUDA — training cross-encoder trên CPU là vô ích. "
                         "Chạy trên máy có GPU.")


    from transformers import AutoConfig
    base_config = AutoConfig.from_pretrained(cfg.model)
    base_config.num_labels = 1
    tok = AutoTokenizer.from_pretrained(cfg.model)
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model, config=base_config)
    model.to(device)
    model.gradient_checkpointing_enable()

    collate = make_collate(tok, cfg)
    dl_tr = DataLoader(GroupDataset(train_rows, cfg.group_size, rng),
                       batch_size=cfg.batch_size, shuffle=True,
                       collate_fn=collate, num_workers=2, drop_last=True)
    dl_va = DataLoader(GroupDataset(val_rows, cfg.group_size, rng),
                       batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)

    steps = len(dl_tr) // cfg.grad_accum * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(steps * cfg.warmup_ratio), steps)

    print(f"device={device} steps={steps} pairs/forward={cfg.pairs_per_forward}", flush=True)
    history: dict[str, list] = {"epoch": [], "train_loss": [], "val_loss": [], "val_top1": []}
    step = 0
    for ep in range(cfg.epochs):
        running = []
        for i, (enc, n) in enumerate(dl_tr):
            enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
            labels = torch.zeros(n, dtype=torch.long, device=device)


            with torch.autocast(device, dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = model(**enc).logits.view(n, cfg.group_size)
                loss = F.cross_entropy(logits, labels) / cfg.grad_accum
            loss.backward()
            running.append(loss.item() * cfg.grad_accum)

            if (i + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    print(f"  ep{ep} step{step}/{steps} loss={np.mean(running[-200:]):.4f} "
                          f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        vl, vacc = evaluate(model, dl_va, device, cfg.group_size)
        tl = float(np.mean(running))
        print(f"[epoch {ep}] train_loss={tl:.4f} val_loss={vl:.4f} "
              f"val_top1_in_group={vacc:.4f}", flush=True)
        history["epoch"].append(ep)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_top1"].append(vacc)
        out = Path(cfg.out) / f"epoch{ep}"
        out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(out); tok.save_pretrained(out)
        (out / "train_config.json").write_text(json.dumps(vars(cfg), indent=2))
        print(f"  saved {out}", flush=True)

    summary = {"config": vars(cfg), "history": history,
               "best_epoch": int(np.argmax(history["val_top1"])),
               "best_val_top1": float(max(history["val_top1"]))}
    p = Path(cfg.summary)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary -> {p}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
