# Bidirectional handwritten English–Chinese translation

Group 10’s Colab project reads handwritten English with TrOCR and Chinese or mixed lines with PP-OCRv5. It translates each reading into English or Simplified Chinese. Uploaded pages show numbered line boxes. Live OCR reads one line inside a fixed camera guide. Readers can review suggested corrections before translating edited text.

## Run the demo

1. Open the [presenter in Colab](https://colab.research.google.com/github/Neoweikang/cvdl_10/blob/main/notebooks/demo.ipynb), select a GPU and choose **Runtime → Run all**.
2. Grant access to the team Drive. The presenter restores the latest completed bilingual run and loads the models. Choose a temporary password and open the presenter link.
3. Sign in as **presenter**. Try the bilingual sample, upload a page, or place a handwritten line inside the live camera guide.

For training, evaluation and saved results, open the [full OCR notebook](notebooks/OCR.ipynb). Choose **restore** to use saved weights or **fresh** to train a new run. The defaults leave the frozen final test off. The [collection notebook](notebooks/data_collection.ipynb) records images and reviewed line labels. See [Wei Kang’s contribution record](docs/contributions.md) for development milestones and commits.

## Historical experiment results

The sections below describe the original repository import. Cell numbers and settings refer to that experiment. The full OCR notebook contains the current results and presentation instructions.

Small, text-free summaries are in `results/ocr/` and `results/translation/`. Values were transcribed from saved notebook outputs, not rerun during repository preparation. Displayed metrics are rounded. No photographs, weights, tokens, source/reference sentences or sentence-level predictions are uploaded.

Public OCR development evaluation uses 64 validation lines per language, with 512 public training lines and seed 42 for shuffled training selection. English CER/WER changed from 1.22%/4.90% to 1.10%/4.71%. Chinese exact/NFKC CER changed from 6.97%/4.61% to 3.84%/3.84%. These are validation subset results, not final test scores. The original private curated snapshot contained 580 English training lines from Wei Kang and no validation/test lines; its adapted checkpoint is unvalidated. Later writer-held-out results are reported in the current OCR notebook.

## Translation development experiments

The original integrated notebook used Cells 19B–19J for the FLORES development experiments. Cell 19B loads `Helsinki-NLP/opus-mt-en-zh` and `Helsinki-NLP/opus-mt-zh-en` in FP32; 19C-D provide basic bidirectional checks. Cell 19E aligns FLORES+ references by sentence ID, selects 200 of 997 development pairs using `random.Random(42).sample(sorted(ids), 200)`, and reuses the same pairs in both directions. Dataset: `openlanguagedata/flores_plus`, revision `1147e2dc59f37fbb5c22fd8478770de02cd399f9`; saved reference fingerprint prefix `e3efbaaa8ed732b6`. devtest was not used.

Cells 19F-G compare beams 1 and 4 on the L4 with batch size 1, three warmups per configuration and three calls per sentence (600 timed calls per configuration). Timing uses `perf_counter` with CUDA synchronisation, includes tokenisation, device transfer, generation and decoding, and excludes loading, warmup, scoring and saving. Decoding is deterministic with no sampling, one returned sequence, maximum 512 input tokens and 256 new output tokens, length penalty 1, cache enabled and early stopping for beam 4. BLEU tokenisation is `zh` for Chinese targets and `13a` for English; chrF uses character order 6, word order 0, beta 2; SacreBLEU 2.6.0.

| Direction | Beams | BLEU | chrF | Median seconds | p95 seconds |
|---|---:|---:|---:|---:|---:|
| English -> Chinese | 1 | 30.303 | 28.025 | 0.145 | 0.256 |
| English -> Chinese | 4 | 31.592 | 29.635 | 0.177 | 0.310 |
| Chinese -> English | 1 | 21.289 | 52.746 | 0.181 | 0.291 |
| Chinese -> English | 4 | 21.175 | 53.175 | 0.225 | 0.356 |

Beam 4 increases latency in both directions. It improves aggregate English-to-Chinese scores, but Chinese-to-English BLEU slightly decreases despite higher chrF. Cell 19H finds higher/lower/tied sentence chrF counts of 92/68/40 for English-to-Chinese and 77/76/47 for Chinese-to-English: more beams do not consistently improve individual translations. There were no empty outputs, output cap hits or output changes across repeated benchmark calls.

Cells 19I-J implement reference-free quantity and repetition checks; the saved output confirms all 11 illustrative assertions passed. Each configuration flags 11/200 outputs for suspected quantity mismatches; English-to-Chinese beam 1 also has one repetition flag. Flags overlap and are suspected problems, not verified translation error rates or controller performance.

Model revision hashes, full reference hash and complete generation defaults are stored in private benchmark JSON files but are not visible in this local notebook's saved output. They are explicitly unavailable in the uploaded metadata and must be exported from Drive before claiming exact reproduction. Those early runs did not export all revision fields. The 200-pair dev set is small; no COMET, human correctness assessment or statistical significance is established. Later translation and presenter work is documented in the contribution record.
