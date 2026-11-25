"""Publish the trained OCR weights and the app to Hugging Face.

1. Log in once:   hf auth login        (paste a WRITE token from huggingface.co/settings/tokens)
2. Upload models: python publish.py models --run-dir PATH_TO_SAVED_RUN --user YOUR_HF_USERNAME --langs en
3. Create Space:  python publish.py space --user YOUR_HF_USERNAME --read-token hf_xxx --baseline zh

--langs en uploads only the trained English model; --baseline zh makes the Space
use the pretrained Chinese model (best on W01 validation).

The model repos are private because the checkpoints were trained on IAM, CASIA
and the team's own handwriting. The Space reads them with a READ token stored as
a Space secret. Rerun `space` after editing app.py or pipeline.py to redeploy.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

HERE = Path(__file__).resolve().parent
MODEL_REPOS = {'en': 'cvdl10-trocr-en', 'zh': 'cvdl10-ppocr-zh'}
# Pretrained foundation models (public), used when a language keeps its baseline.
BASELINE_OCR = {'en': 'microsoft/trocr-base-handwritten',
                'zh': 'PaddlePaddle/PP-OCRv5_server_rec_safetensors'}
SPACE_NAME = 'cvdl10-handwriting-translator'
SPACE_README = """---
title: CVDL Handwriting Translator
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.31.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# Handwriting translator (CVDL Group 10)

Reads one handwritten line in English or Chinese and translates it into the other
language. Runs on the free CPU tier with int8 dynamically quantised models.
Source code: https://github.com/Neoweikang/cvdl_10 (deploy/).
"""


def upload_models(api: HfApi, user: str, run_dir: Path, langs=('en', 'zh')):
    for lang in langs:
        name = MODEL_REPOS[lang]
        folder = run_dir / lang / 'weights'
        if not (folder / 'config.json').is_file():
            sys.exit(f'{folder} has no config.json. --run-dir must contain {lang}/weights.')
        repo_id = f'{user}/{name}'
        api.create_repo(repo_id, repo_type='model', private=True, exist_ok=True)
        print(f'Uploading {folder} -> {repo_id} (private)…', flush=True)
        api.upload_folder(folder_path=str(folder), repo_id=repo_id, repo_type='model',
                          commit_message=f'Upload trained {lang} OCR weights from {run_dir.name}')
    print('Models uploaded.')


def publish_space(api: HfApi, user: str, read_token: str | None, private: bool, baseline=(),
                  hardware: str = 'cpu-basic', sleep_minutes: int = 15):
    repo_id = f'{user}/{SPACE_NAME}'
    if api.repo_exists(repo_id, repo_type='space'):
        print(f'Space {repo_id} exists; updating it.')
    else:
        try:
            extra = {'space_sleep_time': sleep_minutes * 60} if hardware != 'cpu-basic' else {}
            api.create_repo(repo_id, repo_type='space', space_sdk='gradio', space_hardware=hardware,
                            private=private, exist_ok=True, **extra)
        except HfHubHTTPError as error:
            response = getattr(error, 'response', None)
            detail = response.text if response is not None else str(error)
            sys.exit(f'Hugging Face refused to create the Space ({error}).\nServer message: {detail}\n\n'
                     f'Create it in the browser instead: https://huggingface.co/new-space\n'
                     f'  Owner: {user}   Space name: {SPACE_NAME}   SDK: Gradio   Hardware: {hardware}\n'
                     'Then rerun this command; it will upload the app into the existing Space.')
    for local, remote in (('app.py', 'app.py'), ('pipeline.py', 'pipeline.py'),
                          ('requirements-space.txt', 'requirements.txt')):
        api.upload_file(path_or_fileobj=str(HERE / local), path_in_repo=remote,
                        repo_id=repo_id, repo_type='space')
    api.upload_file(path_or_fileobj=SPACE_README.encode('utf-8'), path_in_repo='README.md',
                    repo_id=repo_id, repo_type='space')
    for lang, name in MODEL_REPOS.items():
        source = BASELINE_OCR[lang] if lang in baseline else f'{user}/{name}'
        api.add_space_variable(repo_id, f'{lang.upper()}_OCR', source)
        print(f'{lang.upper()}_OCR = {source}')
    api.add_space_variable(repo_id, 'OPTIMISE', 'int8')
    if hardware != 'cpu-basic':
        api.request_space_hardware(repo_id, hardware, sleep_time=sleep_minutes * 60)
        print(f'Hardware: {hardware}, billed per minute only while Starting/Running; '
              f'sleeps (no billing) after {sleep_minutes} idle minutes.')
    if set(baseline) >= set(MODEL_REPOS):
        print('Both languages use public baseline models; no read token is needed.')
    elif read_token:
        api.add_space_secret(repo_id, 'HF_TOKEN', read_token)
    else:
        print('No --read-token given. Add a READ token as the secret HF_TOKEN in the Space settings, '
              'or the Space cannot load the private models.')
    print(f'Space published: https://huggingface.co/spaces/{repo_id}')
    print('The first build installs packages and downloads about 4 GB of models: allow 10-15 minutes.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    models = sub.add_parser('models', help='Upload trained OCR weights to private model repos')
    models.add_argument('--run-dir', type=Path, required=True, help='Saved training run folder')
    models.add_argument('--user', required=True)
    models.add_argument('--langs', default='en,zh', help='Which trained languages to upload, e.g. en')
    space = sub.add_parser('space', help='Create or update the Gradio Space')
    space.add_argument('--user', required=True)
    space.add_argument('--read-token', help='READ token the Space uses to load the private models')
    space.add_argument('--private', action='store_true', help='Make the Space itself private')
    space.add_argument('--baseline', default='', help='Languages that use the pretrained model, e.g. zh')
    space.add_argument('--hardware', default='cpu-basic', choices=['cpu-basic', 'cpu-upgrade'],
                       help='cpu-upgrade (8 vCPU, 32 GB) is billed per running minute; needs a payment method')
    space.add_argument('--sleep-minutes', type=int, default=15,
                       help='Paid hardware sleeps after this many idle minutes (not billed while asleep)')
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()['name']
    if who != args.user:
        print(f'Note: logged in as {who}, publishing under {args.user}.')
    if args.command == 'models':
        langs = [lang.strip() for lang in args.langs.split(',') if lang.strip()]
        if not langs or set(langs) - set(MODEL_REPOS):
            sys.exit('--langs must be en, zh or en,zh')
        upload_models(api, args.user, args.run_dir.expanduser().resolve(), langs)
    else:
        baseline = [lang.strip() for lang in args.baseline.split(',') if lang.strip()]
        if set(baseline) - set(MODEL_REPOS):
            sys.exit('--baseline must be en, zh or en,zh')
        publish_space(api, args.user, args.read_token, args.private, baseline,
                      args.hardware, args.sleep_minutes)


if __name__ == '__main__':
    main()
