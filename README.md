# LegalIR

Truy hồi văn bản pháp luật tiếng Việt: 4 retriever song song (BM25 + 3 dense) gộp bằng weighted RRF, xếp lại bằng cross-encoder, trả về 5 văn bản cho mỗi câu hỏi.

## 1. Pipeline

```
contexts + queries (có nhãn và chưa gắn nhãn)
        │
        ▼
1. preprocess ──► corpus (retrieval_chunks.jsonl, queries/*.jsonl)
        │
        ▼
2. bm25 ┐
3. vae  │ 4 retriever, mỗi nguồn 200 văn bản/câu
4. bgem3│
5. f2llm┘
        │
        ▼
6. fuse.py ──► fused (weighted RRF, 200 ứng viên/câu)
        │
        ▼
7. rerank.py ──► logit cross-encoder cho pool 200
        │
        ▼
8. apply_ranking_rules.py ──► submission.json (5 context_id/câu)
```

## 2. Code map

### Chạy pipeline

| File | Bước | Việc |
| ---- | ---- | ---- |
| `preprocess.py` | 1 | Dựng corpus từ văn bản gốc, tách đơn vị pháp lý thành chunk, ghi workspace kèm báo cáo chất lượng. Hai lệnh: `build`, `validate`. |
| `index_bm25.py` | 2 | Dựng chỉ mục BM25 chuẩn Lucene (`bm25s`, b=0.55) trên corpus chunk. |
| `retrieve_bm25.py` | 2 | Nạp chỉ mục BM25, xếp hạng theo câu hỏi, ghi `ranked.jsonl`. |
| `index_dense.py` | 3–5 | Encode corpus bằng một encoder (`vae-v2`, `bge-m3`, `f2llm-1.7b`), lưu chỉ mục vector kèm manifest tham số. |
| `retrieve_dense.py` | 3–5 | Encode câu hỏi, tìm chunk gần nhất trong chỉ mục vector, ghi `ranked.jsonl`. |
| `fuse.py` | 6 | Gộp nhiều bảng xếp hạng bằng weighted RRF (`w_i / (rrf_k + rank_i)`), ghi pool ứng viên. |
| `rerank.py` | 7 | Chấm pool ứng viên bằng cross-encoder đã finetune, ghi logit theo từng context; kiểm ngưỡng tổng tham số hệ thống trước khi chạy. |
| `apply_ranking_rules.py` | 8 | Áp hai luật cố định + một luật theo năm dựu trong câu hỏi lên cặp (fusion, cross-encoder), xuất top-200 và `submission.json` 5 id/câu. |

### Thư viện dùng chung

| File | Vai trò |
| ---- | ------- |
| `commons.py` | Đọc/ghi JSON và JSONL an toàn, chuẩn hoá văn bản tiếng Việt, tách văn pháp lý, băm sha256, đếm tham số checkpoint, kiểm tra submission. |

### Huấn luyện lại checkpoint (ngoài đường chạy suy luận)

| File | Việc |
| ---- | ---- |
| `train_dense_embedder.py` | `split` chia train/dev, `pool` dựng tập ứng viên, `filter` lọc cặp train. |
| `dense_data_lib.py` | Kiểm tra và ghép dữ liệu train: qrels có trong corpus, chuẩn hoá thang điểm, lọc cặp theo nhóm dấu. |
| `dense_format_converters.py` | Đổi triplet chung sang định dạng FlagEmbedding và sentence-transformers. |
| `dense_mining_lib.py` | Chọn ứng viên tiêu cực: ưu tiên ứng viên có điểm chuẩn hoá cao, thêm ứng viên ngẫu nhiên với seed cố định. |
| `build_dense_training_data.py` | Chọn positive cho từng câu train bằng một cross-encoder làm thầy. |
| `mine_dense_negatives.py` | Khoét ứng viên tiêu cực từ kết quả truy hồi + ngẫu nhiên, ghi file cặp train. |
| `mine_reranker_pairs.py` | Khoét cặp (câu hỏi, ứng viên đúng, ứng viên sai) cho cross-encoder. |
| `train_bge_embedder.py` | Fine-tune BGE-M3 một epoch qua bộ fine-tune sẵn có của FlagEmbedding. |
| `run_f2llm_epoch.py` | Fine-tune adapter LoRA cho F2LLM. |
| `train_cross_encoder.py` | Huấn luyện cross-encoder xếp lại ứng viên, lưu checkpoint mỗi epoch. |
| `make_checkpoints_sha256sums.py` | Sinh `checkpoints/SHA256SUMS` để kiểm tra toàn vẹn file trọng số. |
| `system_inventory.json` | Tham số từng thành phần và tổng số tham số hệ thống, để bước rerank tự kiểm ngưỡng. |

## 3. Môi trường

Linux, Python 3.12, CUDA 12.8, GPU ≥ 16 GB, bf16.

Hai môi trường Python riêng, vì adapter LoRA của F2LLM chỉ nạp được với `sentence-transformers` 6.x:

| Môi trường | File                     | Dùng cho                   |
| ---------- | ------------------------ | -------------------------- |
| A          | `requirements.txt`       | mọi bước, trừ F2LLM        |
| B          | `requirements-f2llm.txt` | bước 5 và bước train F2LLM |

Cài đặt, mỗi môi trường một lệnh:

```bash
# Môi trường A
python3.12 -m venv .venv-a
. .venv-a/bin/activate
pip install -r requirements.txt

# Môi trường B — torch phải bản CUDA 12.8, dòng pip trong file sẽ ghi đè
python3.12 -m venv .venv-b
. .venv-b/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-f2llm.txt
```

Kiểm tra môi trường đã đúng trước khi chạy:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 4. Model và checkpoint

Repo không kèm checkpoint. Train lại theo mục 7, hoặc trỏ `--model-name` và `CHAMPION_MANIFEST` vào thư mục checkpoint có sẵn.

| Vai trò | Model                                               | Nguồn                                                                                                      |
| ------- | --------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| vae     | `AITeamVN/Vietnamese_Embedding_v2` (không finetune) | HuggingFace, revision `18b44161e041bf1d3a333ab5144b5b7b93f914d2`                                           |
| bgem3   | `BAAI/bge-m3`, finetune epoch 1                     | thư mục checkpoint trỏ bằng `--model-name`                                                                  |
| f2llm   | `codefuse-ai/F2LLM-1.7B`, LoRA epoch 2              | thư mục adapter trỏ bằng `--model-name`; base HuggingFace revision `cb8d3c8032406948af52b960ebb05b9e7830ac3a` |
| rerank  | `BAAI/bge-reranker-v2-m3`, finetune epoch 2         | thư mục trỏ bằng `CHAMPION_MANIFEST`                                                                       |
| bm25    | không dùng model neural                             | —                                                                                                          |

Cần Internet để tải model gốc của vae và base của f2llm. Tải đúng revision đã ghi ở bảng; nếu repository đó có bản mới thì kết quả sẽ lệch.

**Tổng tham số toàn hệ thống: 3.430.262.785**. Con số đếm trực tiếp từ header file `safetensors`, ghi trong `system_inventory.json`. Muốn bước rerank tự kiểm và ghi con số này vào `run_manifest.json`, đặt biến `SYSTEM_INVENTORY=system_inventory.json`; vượt ngưỡng thì job dừng ngay.

## 5. Dữ liệu

**Kho văn bản (`--contexts-dir`).** Thư mục chứa các file `context_*.json`, mỗi file một văn bản (trường `passage` là nội dung, `link` là nguồn gốc). Văn bản nào `passage` rỗng thì preprocess bỏ qua.

**Câu hỏi.**

- File có nhãn (ví dụ `train.json`): `{"123": {"question": "...", "answer": ["45"]}}`.
- File chưa gắn nhãn (ví dụ `queries.json`): `{"123": {"question": "..."}}`.

## 6. Cách chạy

`<W>` = thư mục workspace, `<R>` = thư mục retrieval (chứa `indexes/` và `outputs/`).

```bash
# 1) Preprocess
python preprocess.py build --contexts-dir contexts/corpus \
    --queries train.json queries.json --labeled-queries train.json \
    --output <W>
python preprocess.py validate --workspace <W>
# → <W>/corpus/retrieval_chunks.jsonl, <W>/queries/{train,queries}.jsonl
# --labeled-queries: chỉ file được liệt kê mới đọc trường "answer".

# 2–5) Bốn retriever: dựng index một lần, rồi truy hồi trên tập chưa gắn nhãn.
#      bm25, vae, bgem3 chạy ở môi trường A; f2llm ở môi trường B.
python index_bm25.py --corpus <W>/corpus/retrieval_chunks.jsonl --index-dir <R>/indexes/bm25
python retrieve_bm25.py --index-dir <R>/indexes/bm25 --queries <W>/queries/queries.jsonl \
    --output-dir <R>/outputs/bm25

python index_dense.py --encoder vae-v2 --corpus <W>/corpus/retrieval_chunks.jsonl \
    --index-dir <R>/indexes/vae
python retrieve_dense.py --index-dir <R>/indexes/vae --queries <W>/queries/queries.jsonl \
    --output-dir <R>/outputs/vae

python index_dense.py --encoder bge-m3 --model-name <checkpoint_bge_m3> \
    --corpus <W>/corpus/retrieval_chunks.jsonl --index-dir <R>/indexes/bgem3
python retrieve_dense.py --index-dir <R>/indexes/bgem3 --queries <W>/queries/queries.jsonl \
    --output-dir <R>/outputs/bgem3

python index_dense.py --encoder f2llm-1.7b --model-name <checkpoint_f2llm> \
    --corpus <W>/corpus/retrieval_chunks.jsonl --index-dir <R>/indexes/f2llm
python retrieve_dense.py --index-dir <R>/indexes/f2llm --queries <W>/queries/queries.jsonl \
    --output-dir <R>/outputs/f2llm
# retrieve_dense.py đọc encoder, dims, max_length từ manifest của index, và tự chọn
# chunk-depth theo encoder (vae 2048, bge-m3 và f2llm 8192).

# 6) Fusion 4 nguồn
python fuse.py \
    --source bm25=<R>/outputs/bm25/ranked.jsonl \
    --source vae=<R>/outputs/vae/ranked.jsonl \
    --source bgem3=<R>/outputs/bgem3/ranked.jsonl \
    --source f2llm=<R>/outputs/f2llm/ranked.jsonl \
    --output-dir <R>/outputs/fused

# 7) Rerank cross-encoder (môi trường A, cần GPU)
export PROJECT_ROOT=<gốc project> RETRIEVAL_DIR=<R> QUERY_WORKSPACE=<W>
export QUERIES=<W>/queries/queries.jsonl
export CHAMPION_MANIFEST=<manifest của checkpoint rerank>
RERANK_WORKFLOW=private OUTPUT_DIR=<R>/outputs/rerank python rerank.py
# → <R>/outputs/rerank/signals/ce_context_max.jsonl
# `private` là tên workflow trong rerank.py, override được qua biến QUERIES.

# 8) Xếp hạng cuối (CPU)
python apply_ranking_rules.py apply-ranking-rules \
    --fused <R>/outputs/fused/ranked.jsonl \
    --ce-logits <R>/outputs/rerank/signals/ce_context_max.jsonl \
    --questions queries.json --train-json train.json \
    --corpus <W>/corpus/retrieval_chunks.jsonl --out-dir <out>
# → <out>/submission.json (5 context_id mỗi câu), submission.zip, top200.jsonl, report.json
```

## 7. Train lại checkpoint

Chỉ cần khi muốn dựng lại checkpoint từ đầu. Nhãn train lấy từ file có nhãn (ví dụ `train.json`).

| Bước           | File                             | Vào                                              | Ra                       |
| -------------- | -------------------------------- | ------------------------------------------------ | ------------------------ |
| chia dev       | `train_dense_embedder.py split`  | `train.json`                                     | `dev_split_qids.json`    |
| chọn positive  | `build_dense_training_data.py`   | `train.json`, corpus, một cross-encoder làm thầy | jsonl positive           |
| mine negative  | `mine_dense_negatives.py`        | positives, các `ranked.jsonl`, corpus            | jsonl cặp train          |
| lọc train-only | `train_dense_embedder.py filter` | jsonl cặp train, dev split                       | triplets + flagembedding |
| train F2LLM    | `run_f2llm_epoch.py`             | triplets, `--lr`, `--rank`, `--epoch`            | checkpoint LoRA          |
| train BGE-M3   | `train_bge_embedder.py`          | flagembedding, `--lr`, `--out-dir`               | checkpoint               |
| train reranker | `train_cross_encoder.py`         | jsonl cặp của `mine_reranker_pairs.py`           | checkpoint mỗi epoch     |

`train_bge_embedder.py` gọi bộ fine-tune sẵn có của thư viện FlagEmbedding (`--unified_finetuning`, `--use_self_distill`). Dùng `--dry-run` để xem lệnh trước khi chạy.

## License

MIT — xem [LICENSE](LICENSE).
