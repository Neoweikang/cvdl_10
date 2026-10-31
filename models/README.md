# Models and private checkpoints

Pretrained IDs: `microsoft/trocr-base-handwritten`, `PaddlePaddle/PP-OCRv5_server_rec_safetensors`, `Helsinki-NLP/opus-mt-en-zh`, `Helsinki-NLP/opus-mt-zh-en`. The notebooks download matching processors/tokenizers from Hugging Face. Revisions are not pinned by current loaders; export translation `revision` fields from private benchmark records before reproducing the exact recorded run.

Drive checkpoint root: `/content/drive/MyDrive/handwritten_translation_project/checkpoints/`.

- `english_trocr_public_epoch1`: one epoch on 512 IAM training lines (Cell 11).
- `chinese_ppocr_public_frozen_bn_epoch1`: one epoch on 512 CASIA training lines with fixed BatchNorm statistics (Cell 15).
- `curated/en_v1` or `curated/zh_v1`: optional additional adaptation from public checkpoints (Cell 18); `selected.json` identifies the selected checkpoint folder and training provenance. Only English adaptation appears available in the saved output; it is explicitly unvalidated because no curated validation samples were available.

Obtain checkpoints and their training records from the team Drive. Run Cells 1-6A and 17, then use `load_ocr('en', 'baseline')`, `load_ocr('en', 'public')` or `load_ocr('en', 'adapted')` (use `zh` for Chinese). Missing checkpoints are shown unavailable by the final interface. Translation needs no training: run Cell 19B and `load_mt('en')` or `load_mt('zh')`. No model weights are committed.
