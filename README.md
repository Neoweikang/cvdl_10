# English–Chinese handwriting recognition and translation

CVDL Group 10 built a Colab application that reads handwritten English and Simplified Chinese. Upload a page to see a numbered transcription, or hold one line inside the webcam guide for live translation. The original OCR text stays visible while you review the wording.

## Run the presenter

1. Open the [presenter in Google Colab](https://colab.research.google.com/github/Neoweikang/cvdl_10/blob/main/notebooks/demo.ipynb), select a GPU runtime, and choose **Runtime → Run all**.
2. Allow Colab to mount Google Drive. The presenter loads the pipeline from this repository, restores the latest completed English and Chinese training run, and preloads the models. It requires access to the team's checkpoints at `MyDrive/handwritten_translation_project/saved_training_runs/`.
3. Choose a temporary password when prompted, click **Open handwriting demo**, and sign in as **presenter**. Select **Try a bilingual sample** for a first reading. Keep the Colab runtime connected while presenting.

The saved results are also available to view directly in the [full OCR notebook](notebooks/OCR.ipynb).

## How the pipeline works

1. **Locate the handwriting.** For uploads, the detector straightens the page, suppresses printed notebook lines and draws a numbered box around each detected line. The numbers show the reading order. Check them on two-column pages, where a narrow gap can cause two lines to merge. Live OCR reads only the pixels inside the fixed red guide.
2. **Transcribe each line.** A first reading identifies the script. TrOCR reads English lines, and PP-OCRv5 reads Chinese and mixed-script lines. **Baseline** uses pretrained recognizers. **Trained** applies the project's fine-tuned recognizers to the same crops.
3. **Translate and review.** Marian translates English into Simplified Chinese. NLLB-200 translates Chinese into English and handles mixed-script lines. **Attempt autocorrect** suggests wording you can compare with the handwriting. You can also type a correction. **Translate edits** updates the translation for the lines you changed.

## Saved results

### Handwriting recognition

W01 is a held-out writer who contributed six validation pages. For each language, the pretrained and trained models read the same line crops. Character error rate (CER) counts insertions, deletions and substitutions per 100 reference characters. Lower is better.

| W01 validation set | Lines | Pretrained CER | Trained CER, 18 Nov 2026 |
| --- | ---: | ---: | ---: |
| English | 54 | 17.50% | 6.25% |
| Chinese, reviewed labels | 50 | 4.56% | 5.66% |

Fine-tuning reduced English errors on this writer's pages, while the pretrained Chinese model had the lower error rate. A separate check of the **16 November** checkpoint found 25 character errors across 18 mixed-script CASIA lines, compared with 58 for the baseline. On a two-line mixed-language photo, both made four errors across 35 characters. [Full OCR results and checkpoint comparisons](https://colab.research.google.com/github/Neoweikang/cvdl_10/blob/main/notebooks/OCR.ipynb#scrollTo=checkpoint-ocr-results)

### Translation

The WMT23 benchmark sends typed sentences directly to the translator, isolating translation quality from handwriting errors. BLEU measures overlap with reference word or token sequences. chrF measures character-sequence overlap. Higher scores indicate closer agreement with the references. [WMT23 results in Colab](https://colab.research.google.com/github/Neoweikang/cvdl_10/blob/main/notebooks/OCR.ipynb#scrollTo=section-wmt23)

| WMT23 direction | Sentences | Model | BLEU | chrF |
| --- | ---: | --- | ---: | ---: |
| English → Simplified Chinese | 100 | Marian | 33.67 | 30.78 |
| Chinese → English | 100 | NLLB-200 distilled 600M | 17.48 | 43.67 |

The notebook also includes a W01 experiment comparing translations of the same passages from reviewed text and OCR output. The [results directory](results/) archives the original OCR and FLORES+ experiments. The [original benchmark notes](https://github.com/Neoweikang/cvdl_10/blob/505e11d41327c0898ddd3d97947f138b1aa449cc/README.md#translation-development-experiments) describe how the team timed and scored beam search.

## Training and evaluation

In the [full notebook on Colab](https://colab.research.google.com/github/Neoweikang/cvdl_10/blob/main/notebooks/OCR.ipynb), choose **restore** for the in-notebook demo. Leave **SHOW_DEMO** on and the evaluation switches off. **Fresh** trains both recognizers from their pretrained weights. It uses the writer-labelled crops in `MyDrive/CVDL_Handwriting_Dataset/v1_writers/` and the prepared `kaggle_ocr_v1` and `chinese_essays_grid_v1` folders under `MyDrive/handwritten_translation_project/data/public/`. The 18 November training run used 15,777 public and 608 team-labelled English lines, and 36,278 public and 47 team-labelled Chinese lines.

| Evaluation switch | What it runs |
| --- | --- |
| `RUN_OCR_EVALUATION` | Compares recognizers on fixed IAM English and CASIA Chinese validation lines. |
| `RUN_CURATED_VALIDATION` | Scores the team's held-out W01 handwriting. |
| `RUN_FINAL_TEST` | Runs the frozen W03 test. Enable it after model and evaluation choices are final. |

The [collection notebook](notebooks/data_collection.ipynb) captures pages and saves reviewed line crops and labels. The [contribution record](docs/contributions.md) links development milestones to their commits.
