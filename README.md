# Bidirectional handwritten English-Chinese translation

Group 10 academic prototype: recognise handwritten English or Simplified Chinese, then translate the recognised text into the other language. English uses TrOCR; Chinese uses PP-OCRv5; translation uses bilingual Marian models. The intended pipeline is camera/upload -> cropped line -> OCR -> contextual revision -> English/Chinese translation. Contextual revision and automatic end-to-end translation in the camera interface remain unfinished: the current final interface performs OCR, while translation is called separately.

## Setup and notebook order

Use Google Colab with a GPU and private Google Drive storage. The saved run used Python 3.13.16, NVIDIA L4, torch 2.11.0+cu130 and torchvision 0.26.0+cu130. Keep Colab's matching PyTorch/CUDA installation; install `requirements.txt` with `%pip install -r requirements.txt` or use the numbered install cells. Restart the runtime if replacing an already imported Transformers version. These are recorded notebook dependencies, not a complete environment lock; pandas, Hub and collection dependency versions were not captured. The notebooks require Colab APIs and do not run unchanged as local Jupyter notebooks. No separate incompatible environment is currently specified.

1. Run `notebooks/data_collection.ipynb` in order to capture/upload images, review crops and correct labels before saving. See `data/README.md` for the separate curated writer split.
2. Run `notebooks/prototype.ipynb` Cells 1-16 in order for runtime setup, Drive, dataset verification, public OCR subsets, baselines and one-epoch training. Cell 4A needs the private curated dataset; obtain it from the team first.
3. Run Cells 17-19 for checkpoint loading, optional curated adaptation and held-out comparison. Cell 19A is opt-in final-test evaluation: freeze settings before setting `RUN_FINAL_TEST=True`.
4. Run Cells 19B-19J in order for translation, FLORES development evaluation, beam comparison and quality checks. Obtain FLORES+ access and provide a read token through the hidden prompt or Colab secret; never save it in Git.
5. Run Cell 20 for upload/camera OCR. Select a language and an available checkpoint, upload one cropped line or open the camera, then capture or start live OCR. For translation separately call `translate_text(text, 'en')` or `translate_text(text, 'zh')` after Cell 19C.

Private Drive roots and checkpoint loading are documented in `data/README.md` and `models/README.md`. Original local notebooks remain outside this checkout; submitted copies retain code, numbered cells and comments with outputs cleared.

## Current evidence and limitations

Small, text-free summaries are in `results/ocr/` and `results/translation/`. Values were transcribed from saved notebook outputs, not rerun during repository preparation. Displayed metrics are rounded. No photographs, weights, tokens, source/reference sentences or sentence-level predictions are uploaded.

Public OCR development evaluation uses 64 validation lines per language, with 512 public training lines and seed 42 for shuffled training selection. English CER/WER changed from 1.22%/4.90% to 1.10%/4.71%. Chinese exact/NFKC CER changed from 6.97%/4.61% to 3.84%/3.84%. These are validation subset results, not final test scores. The private curated snapshot contains 580 English training lines from Wei Kang and no validation/test lines; its adapted checkpoint is unvalidated. Final writer-held-out evaluation remains pending.

## Translation development experiments

Translation development remains in the integrated notebook (Cells 19B-19J). Cell 19B loads `Helsinki-NLP/opus-mt-en-zh` and `Helsinki-NLP/opus-mt-zh-en` in FP32; 19C-D provide basic bidirectional checks. Cell 19E aligns FLORES+ references by sentence ID, selects 200 of 997 development pairs using `random.Random(42).sample(sorted(ids), 200)`, and reuses the same pairs in both directions. Dataset: `openlanguagedata/flores_plus`, revision `1147e2dc59f37fbb5c22fd8478770de02cd399f9`; saved reference fingerprint prefix `e3efbaaa8ed732b6`. devtest was not used.

Cells 19F-G compare beams 1 and 4 on the L4 with batch size 1, three warmups per configuration and three calls per sentence (600 timed calls per configuration). Timing uses `perf_counter` with CUDA synchronisation, includes tokenisation, device transfer, generation and decoding, and excludes loading, warmup, scoring and saving. Decoding is deterministic with no sampling, one returned sequence, maximum 512 input tokens and 256 new output tokens, length penalty 1, cache enabled and early stopping for beam 4. BLEU tokenisation is `zh` for Chinese targets and `13a` for English; chrF uses character order 6, word order 0, beta 2; SacreBLEU 2.6.0.

| Direction | Beams | BLEU | chrF | Median seconds | p95 seconds |
|---|---:|---:|---:|---:|---:|
| English -> Chinese | 1 | 30.303 | 28.025 | 0.145 | 0.256 |
| English -> Chinese | 4 | 31.592 | 29.635 | 0.177 | 0.310 |
| Chinese -> English | 1 | 21.289 | 52.746 | 0.181 | 0.291 |
| Chinese -> English | 4 | 21.175 | 53.175 | 0.225 | 0.356 |

Beam 4 increases latency in both directions. It improves aggregate English-to-Chinese scores, but Chinese-to-English BLEU slightly decreases despite higher chrF. Cell 19H finds higher/lower/tied sentence chrF counts of 92/68/40 for English-to-Chinese and 77/76/47 for Chinese-to-English: more beams do not consistently improve individual translations. There were no empty outputs, output cap hits or output changes across repeated benchmark calls.

Cells 19I-J implement reference-free quantity and repetition checks; the saved output confirms all 11 illustrative assertions passed. Each configuration flags 11/200 outputs for suspected quantity mismatches; English-to-Chinese beam 1 also has one repetition flag. Flags overlap and are suspected problems, not verified translation error rates or controller performance.

Model revision hashes, full reference hash and complete generation defaults are stored in private benchmark JSON files but are not visible in this local notebook's saved output. They are explicitly unavailable in the uploaded metadata and must be exported from Drive before claiming exact reproduction. Current model loading does not pin revisions. The 200-pair dev set is small; no COMET, human correctness assessment or statistical significance is established. Adaptive controller, COMET, additional model comparison, contextual revision, deployment, final poster and final report remain pending. Add final submission artifacts and updated demo instructions when ready.
