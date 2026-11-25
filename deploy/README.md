# Deployment

The handwriting translator runs as the same Gradio app in two places:

| | Local | Cloud |
|---|---|---|
| Where | Your computer (`python app.py`) | Modal, free Starter credits (`modal deploy modal_app.py`) |
| Address | http://127.0.0.1:7860 | https://scribeten--cvdl10-handwriting-translator-web.us-east.modal.run |
| Hardware | CPU, or NVIDIA GPU with CUDA PyTorch | 4 CPU cores, 12 GB memory, scales to zero after 10 idle minutes |
| Optimisation | `OPTIMISE_SCOPE=mt`: OCR in FP32, translation int8 (CPU) or FP16 (GPU) | OCR in FP32, translation int8 |
| English OCR | Trained TrOCR, 9 Nov run (`20261109T033914Z_187fd23d`) | Same weights, private Hub repo `reinertong/cvdl10-trocr-en` |
| Chinese OCR | Pretrained PP-OCRv5 server recogniser | Same |
| Translation | Marian opus-mt-en-zh (EN→ZH), NLLB-200 distilled 600M (ZH→EN) | Same |

Scope: one handwritten line per request (upload, paste or webcam photo). Page detection,
line segmentation, live camera reading and autocorrect remain in the Colab presenter
(`notebooks/demo.ipynb`) and `notebooks/OCR.ipynb`.

## Files

| File | Purpose |
|---|---|
| `pipeline.py` | Model loading, optimisation, OCR, language routing and translation. Mirrors the notebook's checkpoint loading, `transcribe`, translation routes and script detection. |
| `app.py` | Gradio interface styled after the Colab presenter: upload, webcam or clipboard; auto or manual language; editable transcription; per-stage timings. |
| `modal_app.py` | Cloud deployment on Modal: container image, model cache volume, Hugging Face secret, scaling settings. |
| `benchmark.py` | Compares FP32 with int8/FP16: weight size, start-up, OCR and translation latency, CER. |
| `publish.py` | Uploads trained OCR weights to private Hugging Face model repos (`models`). Its `space` command targets Hugging Face Spaces, which now need a paid plan for Gradio apps. |
| `requirements.txt` | Local packages (install PyTorch separately first). |
| `requirements-space.txt` | Hugging Face Space packages (kept for reference; the Space route needs PRO). |

## 1. Model choice

Checkpoints were chosen by W01 validation CER (notebook "Checkpoint results"), never by test writers:

| W01 CER | Baseline | 9 Nov | 16 Nov | 18 Nov |
|---|---:|---:|---:|---:|
| English (54 lines) | 17.50% | **5.00%** | 5.54% | 6.25% |
| Chinese (50 lines) | **4.56%** | 5.47% | 5.29% | 5.66% |

Deployed: **trained English from the 9 Nov run** and the **pretrained Chinese baseline**.
Download `MyDrive/handwritten_translation_project/saved_training_runs/20261109T033914Z_187fd23d`
from the team Drive. It must contain `en/weights/config.json`.

## 2. Optimisation finding

int8 dynamic quantisation converts every `nn.Linear` layer to 8-bit weights. On the same
handwritten line with the same 9 Nov English model:

| Precision | Reading | Confidence |
|---|---|---:|
| FP32 | Hello my name is Reiner | 0.98 |
| int8 (local CPU) | thella any name is beiner | 0.17 |
| int8 (Modal CPU) | the 1/2mq name is Reimer | 0.19 |

TrOCR is almost entirely linear layers and decodes one token at a time, so rounding errors
compound along the line. Both OCR models therefore stay in FP32 (`OPTIMISE_SCOPE=mt`) and only
Marian and NLLB are quantised. Run `benchmark.py` on W01 lines to measure the effect as CER.

## 3. Set up (Windows PowerShell)

```powershell
cd cvdl_10\deploy
py -3.14 -m venv .venv          # Python 3.11 to 3.14 work
.\.venv\Scripts\Activate.ps1    # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install torch torchvision   # CPU build; for an NVIDIA GPU use the CUDA command from pytorch.org
pip install -r requirements.txt
```

## 4. Run locally

```powershell
$env:TRAINED_DIR = "C:\path\to\20261109T033914Z_187fd23d"
$env:ZH_OCR = "PaddlePaddle/PP-OCRv5_server_rec_safetensors"   # pretrained Chinese
$env:OPTIMISE_SCOPE = "mt"                                       # OCR in FP32
python app.py
```

Open the printed URL (normally http://127.0.0.1:7860). The first start downloads Marian,
NLLB and the tokenizers (about 3 GB) into the Hugging Face cache; later starts take seconds.
Webcam: click the camera button to take the photo, then **Read and translate**.

## 5. Deploy to the cloud (Modal)

1. Upload the trained English weights to a private Hugging Face repo (once):
   ```powershell
   hf auth login        # WRITE token from huggingface.co/settings/tokens
   python publish.py models --run-dir "C:\path\to\20261109T033914Z_187fd23d" --user YOUR_HF_USERNAME --langs en
   ```
2. Log in to Modal and store a Hugging Face **read** token as a secret (once):
   ```powershell
   pip install modal
   modal setup
   modal secret create cvdl10-hf HF_TOKEN=hf_YOUR_READ_TOKEN
   ```
3. Deploy, and redeploy after any code change (same URL):
   ```powershell
   modal deploy modal_app.py
   ```

The first start downloads about 4 GB of models into a Modal volume (5 to 10 minutes).
Later cold starts load from the volume in 1 to 3 minutes. Before a demo, keep one container
warm with `$env:CVDL_WARM = "1"; modal deploy modal_app.py`, and set it back to `"0"`
afterwards. Without a payment method the Starter plan gives US$1 of credit per month;
at about US$0.29 per running hour, that covers testing and a demo.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRAINED_DIR` | unset | Saved run folder containing `en/weights` (and `zh/weights`) |
| `EN_OCR`, `ZH_OCR` | from `TRAINED_DIR`, else the foundation models | Local folder or Hub repo id; overrides `TRAINED_DIR` |
| `DEVICE` | `cuda` if available, else `cpu` | Force `cpu` to test the cloud setup locally |
| `OPTIMISE` | `auto` (`fp16` on GPU, `int8` on CPU) | `none`, `fp16` or `int8` |
| `OPTIMISE_SCOPE` | `all` locally, `mt` on Modal | `mt` keeps OCR in FP32 and optimises translation only |
| `HF_TOKEN` | unset | Read token for the private English model repo |
| `CVDL_WARM` | `0` | Modal only: `1` keeps one container running |

## Design notes and limitations

- **Decoding matches the notebook.** Greedy TrOCR with 128 new tokens, CTC collapse for
  PP-OCRv5, Marian with `>>cmn_Hans<<` and NLLB with `zho_Hans → eng_Latn`, 4 beams,
  256 new tokens. Model revisions are pinned to those the notebook recorded.
- **Language routing.** Auto mode reads with PP-OCRv5 first, because its vocabulary covers
  both scripts, and rereads with TrOCR when Latin letters outnumber Han characters.
  Choosing the language manually saves one OCR pass.
- **Confidence** is the lower-quintile token probability, the same measure as the
  notebook's `transcribe(..., with_scores=True)`. The app shows the raw score without
  a warning threshold, because the notebook's thresholds are calibrated per checkpoint.
- **Cloud hosting.** Hugging Face Spaces now require a PRO subscription for Gradio apps,
  and Vercel's free functions allow 2 GB memory and a 500 MB Python bundle, too small for
  PyTorch plus these models. Modal's free CPU credits host the app instead.
- **CPU speed.** English OCR takes about 1.5 to 2.3 s per line on CPU, which is why the
  deployed app reads photos rather than a live camera stream.
- **`torch.ao.quantization` is deprecated** in recent PyTorch. It works in the pinned
  2.15.0; a future migration path is `torchao`.
- **Data and licences.** The trained OCR weights stay in a private repo because they were
  trained on IAM, CASIA and the team's own handwriting. NLLB-200 is CC-BY-NC-4.0
  (non-commercial use only). The app keeps images in memory and does not store them.
