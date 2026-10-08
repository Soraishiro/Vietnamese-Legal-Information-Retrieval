# Vietnamese Legal Information Retrieval

Pipeline truy hồi văn bản pháp luật tiếng Việt gồm preprocess, BM25, dense retrieval, fusion, reranking và xuất submission.

## Cấu trúc

```text
src/legal_ir/
├── core/           # I/O, chuẩn hoá văn bản, hashing, index helpers
├── preprocessing/  # dựng và kiểm tra corpus/query
├── retrieval/      # BM25, dense, fusion, rerank, ranking rules
├── training/       # dữ liệu train, mining và fine-tuning
└── tools/          # công cụ quản lý checkpoint
```

Root không chứa implementation Python. Mọi module được chạy qua package `legal_ir`.

## Cài đặt

Yêu cầu Python 3.12. Cài package ở chế độ editable:

```powershell
python -m pip install -e .
python -m legal_ir.preprocessing.preprocess --help
```

Môi trường A dùng cho pipeline chính và hầu hết bước train:

```powershell
python -m venv .venv-a
.venv-a\Scripts\Activate.ps1
pip install -r requirements.txt
```

Môi trường B dùng cho F2LLM:

```powershell
python -m venv .venv-b
.venv-b\Scripts\Activate.ps1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-f2llm.txt
```

## Pipeline

Ký hiệu: `<W>` là workspace corpus, `<R>` là thư mục retrieval output.

### 1. Preprocess

```powershell
python -m legal_ir.preprocessing.preprocess build `
  --contexts-dir contexts/corpus `
  --queries train.json queries.json `
  --labeled-queries train.json `
  --output <W>

python -m legal_ir.preprocessing.preprocess validate --workspace <W>
```

### 2. BM25

```powershell
python -m legal_ir.retrieval.index_bm25 `
  --corpus <W>/corpus/retrieval_chunks.jsonl `
  --index-dir <R>/indexes/bm25

python -m legal_ir.retrieval.retrieve_bm25 `
  --index-dir <R>/indexes/bm25 `
  --queries <W>/queries/queries.jsonl `
  --output-dir <R>/outputs/bm25
```

### 3. Dense retrieval

```powershell
python -m legal_ir.retrieval.index_dense --encoder vae-v2 `
  --corpus <W>/corpus/retrieval_chunks.jsonl `
  --index-dir <R>/indexes/vae

python -m legal_ir.retrieval.retrieve_dense `
  --index-dir <R>/indexes/vae `
  --queries <W>/queries/queries.jsonl `
  --output-dir <R>/outputs/vae
```

Dùng cùng hai module trên cho `bge-m3` và `f2llm-1.7b`, thêm `--model-name` checkpoint tương ứng.

### 4. Fusion, rerank và submission

```powershell
python -m legal_ir.retrieval.fuse `
  --source bm25=<R>/outputs/bm25/ranked.jsonl `
  --source vae=<R>/outputs/vae/ranked.jsonl `
  --source bgem3=<R>/outputs/bgem3/ranked.jsonl `
  --source f2llm=<R>/outputs/f2llm/ranked.jsonl `
  --output-dir <R>/outputs/fused

$env:PROJECT_ROOT = "<project-root>"
$env:RETRIEVAL_DIR = "<R>"
$env:QUERY_WORKSPACE = "<W>"
$env:QUERIES = "<W>/queries/queries.jsonl"
$env:CHAMPION_MANIFEST = "<reranker-manifest>"
$env:RERANK_WORKFLOW = "private"
$env:OUTPUT_DIR = "<R>/outputs/rerank"
python -m legal_ir.retrieval.rerank

python -m legal_ir.retrieval.apply_ranking_rules apply-ranking-rules `
  --fused <R>/outputs/fused/ranked.jsonl `
  --ce-logits <R>/outputs/rerank/signals/ce_context_max.jsonl `
  --questions queries.json `
  --train-json train.json `
  --corpus <W>/corpus/retrieval_chunks.jsonl `
  --out-dir out
```

### 5. Training

Các entrypoint training nằm trong `legal_ir.training`:

```powershell
python -m legal_ir.training.train_dense_embedder --help
python -m legal_ir.training.build_dense_training_data --help
python -m legal_ir.training.mine_dense_negatives --help
python -m legal_ir.training.mine_reranker_pairs --help
python -m legal_ir.training.train_bge_embedder --help
python -m legal_ir.training.run_f2llm_epoch --help
python -m legal_ir.training.train_cross_encoder --help
```

Tạo checksum checkpoint:

```powershell
python -m legal_ir.tools.make_checkpoints_sha256sums
```

## Ghi chú

- Các bước dense và rerank cần GPU, model checkpoint và Internet để tải model gốc.
- F2LLM dùng môi trường Python riêng theo `requirements-f2llm.txt`.
- Dữ liệu, index, output và checkpoint cục bộ đã được loại khỏi Git qua `.gitignore`.

## License

MIT. Xem [LICENSE](LICENSE).