"""Compare unoptimised and optimised models: size, start-up, latency and accuracy.

Usage
  python benchmark.py --data lines.csv
  python benchmark.py --data lines.csv --routing
  python benchmark.py --data lines.csv --configs cpu:none,cpu:int8,cuda:none,cuda:fp16
  python benchmark.py                  # no labelled data: size and latency only

lines.csv columns: image_path,text,language   (language is en or zh; paths may be
relative to the CSV). Use held-out lines only (validation writers, not training).

Writes only aggregate numbers to --out (default results/deployment/). Predictions
are not saved, so no handwriting text from the private dataset ends up in Git.
"""
from __future__ import annotations

import argparse
import gc
import json
import unicodedata
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

from helpers import edit_distance, summarise_ms
from benchmark_data import load_rows
from inference import run_line
from pipeline import Pipeline, Settings, resolve_optimisation
from routing_eval import evaluate_routing

SYNTHETIC_LINES = [
    'The quick brown fox jumps over the lazy dog',
    'Handwriting recognition on a webcam frame',
    'Please meet me at the library at three',
]
SYNTHETIC_MT = {'en': ['I will meet you at the station tomorrow.', 'The weather is cold today.'],
                'zh': ['我明天在车站等你。', '今天天气很冷。']}


def synthetic_rows() -> list[dict]:
    rows = []
    for text in SYNTHETIC_LINES:
        image = Image.new('RGB', (900, 90), 'white')
        try:
            font = ImageFont.load_default(size=40)
        except TypeError:
            font = ImageFont.load_default()
        ImageDraw.Draw(image).text((20, 20), text, fill='black', font=font)
        for lang in ('en', 'zh'):  # Latency only: Chinese OCR also reads this image.
            rows.append(dict(image=image, text=None, language=lang))
    return rows


def run_config(base: Settings, device: str, mode: str, rows: list[dict], repeats: int,
               measure_routing: bool = False) -> dict:
    settings = replace(base, device=device, optimisation=resolve_optimisation(mode, device))
    pipeline = Pipeline(settings).load(warmup=True)
    record = dict(config=f'{device}:{settings.optimisation}', load_seconds=round(pipeline.load_seconds, 1),
                  size_mb={k: round(v, 1) for k, v in pipeline.sizes_mb.items()},
                  size_total_mb=round(sum(pipeline.sizes_mb.values()), 1), languages={})
    labelled_rows = [row for row in rows if row['text'] is not None]
    if measure_routing:
        def route_row(row):
            image = row['image'] if 'image' in row else Image.open(row['image_path']).convert('RGB')
            result = run_line(image, 'auto', pipeline._timed, pipeline.recognise,
                              lambda _text, _language: '')
            return result.language

        record['routing'] = evaluate_routing(labelled_rows, route_row)
    for lang in ('en', 'zh'):
        subset = [r for r in rows if r['language'] == lang]
        if not subset:
            continue
        ocr_ms, mt_ms, errors, errors_nfkc, characters = [], [], 0, 0, 0
        for row in subset:
            image = row['image'] if 'image' in row else Image.open(row['image_path']).convert('RGB')
            prediction = None
            for _ in range(repeats):
                (prediction, _), ms = pipeline._timed(pipeline.recognise, image, lang)
                ocr_ms.append(ms)
            if row['text'] is not None:
                reference = row['text']
                errors += edit_distance(prediction, reference)
                errors_nfkc += edit_distance(unicodedata.normalize('NFKC', prediction),
                                             unicodedata.normalize('NFKC', reference))
                characters += len(reference)
                source = prediction or reference
            else:
                source = SYNTHETIC_MT[lang][len(mt_ms) % len(SYNTHETIC_MT[lang])]
            for _ in range(repeats):
                _, ms = pipeline._timed(pipeline.translate, source, lang)
                mt_ms.append(ms)
        ocr_median, ocr_p95 = summarise_ms(ocr_ms)
        mt_median, mt_p95 = summarise_ms(mt_ms)
        record['languages'][lang] = dict(
            lines=len(subset), ocr_median_ms=round(ocr_median, 1), ocr_p95_ms=round(ocr_p95, 1),
            mt_median_ms=round(mt_median, 1), mt_p95_ms=round(mt_p95, 1),
            cer_percent=round(100 * errors / characters, 2) if characters else None,
            cer_nfkc_percent=round(100 * errors_nfkc / characters, 2) if characters else None,
            reference_characters=characters)
    pipeline.unload()
    del pipeline
    gc.collect()
    return record


def markdown(records: list[dict], labelled: bool) -> str:
    lines = ['| Config | Weights (MB) | Start-up (s) | Language | OCR median / p95 (ms) | '
             'Translate median / p95 (ms) | CER % | CER NFKC % |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for r in records:
        for lang, m in r['languages'].items():
            cer = '–' if m['cer_percent'] is None else f"{m['cer_percent']:.2f}"
            nfkc = '–' if m['cer_nfkc_percent'] is None else f"{m['cer_nfkc_percent']:.2f}"
            lines.append(f"| {r['config']} | {r['size_total_mb']:,.0f} | {r['load_seconds']:.0f} | {lang} | "
                         f"{m['ocr_median_ms']:,.0f} / {m['ocr_p95_ms']:,.0f} | "
                         f"{m['mt_median_ms']:,.0f} / {m['mt_p95_ms']:,.0f} | {cer} | {nfkc} |")
    lines += ['', '| Config | ' + ' | '.join(records[0]['size_mb']) + ' |',
              '|---|' + '---:|' * len(records[0]['size_mb'])]
    for r in records:
        lines.append(f"| {r['config']} | " + ' | '.join(f'{v:,.0f} MB' for v in r['size_mb'].values()) + ' |')
    if labelled and any('routing' in r for r in records):
        lines += ['', '### Automatic language routing',
                  'Expected route is the majority script in the reference text; ties count as Chinese.',
                  '', '| Config | Expected language | Lines | Correct | Accuracy % |',
                  '|---|---|---:|---:|---:|']
        for r in records:
            routing = r.get('routing')
            if not routing:
                continue
            for language in ('en', 'zh'):
                item = routing['by_expected_language'][language]
                accuracy = '–' if item['accuracy_percent'] is None else f"{item['accuracy_percent']:.2f}"
                lines.append(f"| {r['config']} | {language} | {item['lines']} | {item['correct']} | {accuracy} |")
            accuracy = '–' if routing['accuracy_percent'] is None else f"{routing['accuracy_percent']:.2f}"
            lines.append(f"| {r['config']} | **overall** | **{routing['lines']}** | "
                         f"**{routing['correct']}** | **{accuracy}** |")
    if not labelled:
        lines += ['', 'Synthetic printed lines: latency and size only, no accuracy.']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, help='CSV of held-out labelled lines')
    parser.add_argument('--routing', action='store_true',
                        help='also score auto language routing against reference-text script (requires --data)')
    parser.add_argument('--configs', help='Comma-separated device:mode, e.g. cpu:none,cpu:int8')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--out', type=Path, default=Path('results/deployment'))
    args = parser.parse_args()
    if args.routing and not args.data:
        parser.error('--routing requires --data with labelled lines')

    configs = args.configs or ('cpu:none,cpu:int8' + (',cuda:none,cuda:fp16' if torch.cuda.is_available() else ''))
    pairs = [tuple(item.strip().split(':')) for item in configs.split(',')]
    rows = load_rows(args.data) if args.data else synthetic_rows()
    base = Settings.from_env()

    records = []
    for device, mode in pairs:
        print(f'Benchmarking {device}:{mode} on {len(rows)} lines…', flush=True)
        records.append(run_config(base, device, mode, rows, args.repeats, measure_routing=args.routing))

    table = markdown(records, labelled=bool(args.data))
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    meta = dict(created=stamp, torch=torch.__version__,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                en_ocr=base.en_ocr, zh_ocr=base.zh_ocr, data=str(args.data) if args.data else 'synthetic',
                repeats=args.repeats, records=records)
    (args.out / f'benchmark_{stamp}.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    (args.out / f'benchmark_{stamp}.md').write_text(table + '\n', encoding='utf-8')
    print('\n' + table)
    print(f'\nSaved to {args.out.resolve()}')


if __name__ == '__main__':
    main()
