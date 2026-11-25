"""Cloud deployment on Modal (free Starter credits, CPU only).

One-time setup (in the deploy folder, with the venv active):
    pip install modal
    modal setup                                   # browser login
    modal secret create cvdl10-hf HF_TOKEN=hf_YOUR_READ_TOKEN

Deploy (prints the public https://...modal.run URL):
    modal deploy modal_app.py

Before a demo, keep one container warm so nobody waits for a cold start:
    $env:CVDL_WARM = "1"; modal deploy modal_app.py
Afterwards, let it scale to zero again (no cost while idle):
    $env:CVDL_WARM = "0"; modal deploy modal_app.py

The app code is the same app.py and pipeline.py that run locally.
"""
import os

import modal

# Models: trained English (9 Nov run, private Hub repo) + pretrained Chinese baseline,
# chosen by W01 validation CER. Override with EN_OCR / ZH_OCR at deploy time if needed.
EN_OCR = os.environ.get('EN_OCR', 'reinertong/cvdl10-trocr-en')
ZH_OCR = os.environ.get('ZH_OCR', 'PaddlePaddle/PP-OCRv5_server_rec_safetensors')
WARM = int(os.environ.get('CVDL_WARM', '0'))   # read on your PC when you run `modal deploy`
# int8 dynamic quantisation broke TrOCR (FP32 read "Hello my name is Reiner" at confidence 0.98;
# int8 read "thella any name is beiner" at 0.17), so OCR stays FP32 and only translation is int8.
SCOPE = os.environ.get('OPTIMISE_SCOPE', 'mt')

CACHE = '/cache/huggingface'  # Persistent volume: models download once, not on every cold start.

image = (
    modal.Image.debian_slim(python_version='3.12')
    .pip_install('torch==2.15.0', 'torchvision==0.29.2',
                 extra_index_url='https://download.pytorch.org/whl/cpu')
    .pip_install('transformers==5.17.0', 'gradio==6.31.0', 'sentencepiece==0.2.1',
                 'sacremoses==0.1.1', 'Pillow>=11', 'fastapi[standard]')
    .env({'EN_OCR': EN_OCR, 'ZH_OCR': ZH_OCR, 'DEVICE': 'cpu', 'OPTIMISE': 'int8', 'OPTIMISE_SCOPE': SCOPE,
          'HF_HOME': CACHE, 'GRADIO_ANALYTICS_ENABLED': 'False'})
    .add_local_python_source('pipeline', 'app')
)

hf_cache = modal.Volume.from_name('cvdl10-hf-cache', create_if_missing=True)
app = modal.App('cvdl10-handwriting-translator', image=image)


@app.function(
    cpu=4.0,                      # physical cores
    memory=12288,                 # MiB; int8 conversion briefly holds FP32 and int8 copies
    secrets=[modal.Secret.from_name('cvdl10-hf', required_keys=['HF_TOKEN'])],
    volumes={CACHE: hf_cache},
    min_containers=WARM,
    max_containers=1,             # Gradio keeps session state in one process
    scaledown_window=10 * 60,     # idle minutes before scaling to zero
    startup_timeout=15 * 60,      # first start downloads about 4 GB of models
    timeout=10 * 60,
)
@modal.concurrent(max_inputs=20)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app

    import app as ui              # loads and optimises the models (pipeline.Pipeline.load)
    hf_cache.commit()             # keep downloaded models for the next cold start
    return mount_gradio_app(app=FastAPI(), blocks=ui.demo.queue(max_size=16), path='/',
                            theme=ui.THEME, css=ui.CSS, head=ui.HEAD)
