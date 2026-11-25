"""Handwriting translator: deployed app for CVDL Group 10.

Local:  python app.py            (GPU uses FP16, CPU uses int8 by default)
Cloud:  modal deploy modal_app.py  (runs this same file on Modal's CPU)

Configuration comes from environment variables; see deploy/README.md.
The visual design follows the team's Colab presenter (notebooks/demo.ipynb):
paper and ink palette, rust accent, Baskervville headings and DM Sans text.
"""
import html
import inspect

import gradio as gr

from pipeline import LANGUAGE_NAMES, Pipeline, Settings

PIPELINE = Pipeline(Settings.from_env())
print('Loading models…', flush=True)
PIPELINE.load()
print('Ready.', PIPELINE.describe().replace('\n\n', ' | '), flush=True)

LANGUAGE_CHOICES = {'Auto-detect': 'auto', 'English': 'en', 'Chinese': 'zh'}
TARGET = {'en': 'Chinese (Simplified)', 'zh': 'English'}

# --------------------------------------------------------------------------- #
# Design
# --------------------------------------------------------------------------- #
INK, MUTED, RULE, PAPER, PAGE, ACCENT = '#242620', '#4c4e45', '#d5d3c7', '#fdfcf8', '#f7f5ef', '#973d27'
ACCENT_HOVER, STAGE, PANEL_TOP = '#7d311f', '#f5f3eb', '#424839'

_THEME_VARS = dict(
    body_background_fill=PAGE, body_text_color=INK, body_text_color_subdued=MUTED,
    background_fill_primary=PAPER, background_fill_secondary=STAGE,
    border_color_primary=RULE, block_background_fill=PAPER, block_border_color=RULE,
    block_border_width='1px', block_radius='2px', block_shadow='none',
    block_label_text_color=MUTED, block_title_text_color=INK, block_label_background_fill=PAPER,
    input_background_fill='#ffffff', input_border_color=RULE, input_border_color_focus=ACCENT,
    button_primary_background_fill=ACCENT, button_primary_background_fill_hover=ACCENT_HOVER,
    button_primary_text_color='#ffffff',
    button_secondary_background_fill=PAPER, button_secondary_background_fill_hover=STAGE,
    button_secondary_text_color=INK, button_secondary_border_color=RULE,
    color_accent=ACCENT, color_accent_soft='#f3e4df',
    checkbox_background_color='#ffffff', checkbox_border_color='#a9a698',
    checkbox_background_color_selected=ACCENT, checkbox_border_color_selected=ACCENT,
    checkbox_label_background_fill=PAPER, checkbox_label_background_fill_selected='#f3e4df',
    checkbox_label_text_color_selected=INK,
    table_border_color=RULE, table_even_background_fill=PAPER, table_odd_background_fill='#ffffff',
    link_text_color=ACCENT, shadow_drop='none', code_background_fill=STAGE,
)
# One look for everyone: dark-mode browsers get the same paper palette.
# Only colour settings have *_dark variants, so mirror just the ones Gradio defines.
_SETTABLE = set(inspect.signature(gr.themes.Base.set).parameters)
THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont('DM Sans'), 'system-ui', 'sans-serif'],
    font_mono=['ui-monospace', 'Consolas', 'monospace'],
).set(**_THEME_VARS, **{f'{k}_dark': v for k, v in _THEME_VARS.items() if f'{k}_dark' in _SETTABLE})

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Baskervville:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap">')

CSS = f"""
/* Literal colours only: Gradio's CSS processor expands border shorthands and drops var() values. */
body, gradio-app, .gradio-container {{ background:{PAGE} !important; color:{INK}; }}
.gradio-container {{ max-width:1320px !important; margin:0 auto !important; }}
footer {{ display:none !important; }}

/* masthead */
.mast {{ display:flex; align-items:center; justify-content:space-between; min-height:64px;
  border-bottom:1px solid #34372d; margin-bottom:28px; }}
.identity {{ display:flex; align-items:center; gap:12px; font-weight:500; color:{INK}; }}
.seal {{ display:grid; place-items:center; width:35px; height:35px; background:{ACCENT}; color:#ffffff;
  border-radius:3px; font:26px/1 'Noto Serif CJK SC','Songti SC','SimSun',serif; }}
.mast-slash {{ color:{MUTED}; margin:0 6px; }}
.pair {{ color:{INK}; font-weight:500; }}
.pair .arrow {{ color:{ACCENT}; margin:0 4px; }}

/* intro */
.intro h1 {{ font:500 clamp(42px,5.45vw,74px)/1.05 Baskervville, Georgia, 'Times New Roman', serif !important;
  letter-spacing:-.055em; color:{INK}; margin:0 0 14px !important; }}
.intro h1 .dot {{ color:{ACCENT}; }}
.intro p {{ color:{MUTED}; font-size:17px; margin:0; }}
.model-line {{ color:{MUTED}; font-size:13.5px; margin:14px 0 22px; line-height:1.6; }}
.model-line code {{ background:#f0ede3; color:{INK}; padding:1px 6px; border-radius:3px; font-size:12.5px; }}

/* the two panels: the only columns inside the single unequal-height row
   (Gradio 6.31 does not put elem_classes on columns) */
.row.unequal-height > .column {{ background:{PAPER} !important; border-style:solid !important;
  border-color:{RULE} !important; border-width:3px 1px 1px 1px !important; border-top-color:{PANEL_TOP} !important;
  border-radius:2px !important; padding:4px 18px 20px !important; }}
.row.unequal-height > .column:last-child textarea {{ font-size:17px !important; line-height:1.55 !important; }}
.panel-title h2 {{ font:500 clamp(26px,2.3vw,34px)/1.15 Baskervville, Georgia, 'Times New Roman', serif !important;
  letter-spacing:-.045em; color:{INK}; margin:14px 0 6px !important; }}
.panel-note {{ color:{MUTED}; font-size:13px; margin:0 0 10px; }}
.stage {{ border-style:dashed !important; border-width:1px !important; border-color:#bbb9a9 !important;
  border-radius:3px !important; background-color:{STAGE} !important;
  background-image:linear-gradient(#dcd9cc52 1px,transparent 1px),linear-gradient(90deg,#dcd9cc52 1px,transparent 1px) !important;
  background-size:22px 22px !important; }}
.primary-action {{ border-radius:4px !important; font-weight:650 !important; }}
.timing table {{ font-size:13px; }}
.statusbar {{ display:flex; gap:9px; align-items:baseline; color:{MUTED}; font-size:13.5px; margin-top:16px; }}
.statusbar .dot {{ width:8px; height:8px; border-radius:50%; background:#6b7a4f; display:inline-block; flex:none; }}
"""

# --------------------------------------------------------------------------- #
# Callbacks (unchanged behaviour)
# --------------------------------------------------------------------------- #
def _timing_table(timings: dict[str, float]) -> str:
    labels = {'ocr_zh': 'Chinese OCR', 'ocr_en': 'English OCR', 'translate': 'Translation'}
    rows = [f'| {labels.get(k, k)} | {v:,.0f} ms |' for k, v in timings.items()]
    rows.append(f'| **Total** | **{sum(timings.values()):,.0f} ms** |')
    return '| Stage | Time |\n|---|---:|\n' + '\n'.join(rows)


def read_and_translate(image, language_label):
    if image is None:
        raise gr.Error('No photo yet. Using the webcam? Click the camera button to take the photo, '
                       'then press Read and translate.')
    try:
        result = PIPELINE.read_line(image, LANGUAGE_CHOICES[language_label])
    except ValueError as error:
        raise gr.Error(str(error))
    if not result.text:
        raise gr.Error('No text was recognised. Crop to one line of dark writing and try again.')
    confidence = ('n/a' if result.confidence is None else f'{result.confidence:.2f}')
    detected = (f'{LANGUAGE_NAMES[result.language]} ({result.routing}) · '
                f'confidence {confidence}')
    translation_label = f'Translation into {TARGET[result.language]}'
    return (detected, result.text, gr.update(value=result.translation, label=translation_label),
            _timing_table(result.timings_ms), result.language)


def retranslate(text, language):
    if not language:
        raise gr.Error('Read an image first so the source language is known.')
    if not text or not text.strip():
        raise gr.Error('The transcription is empty.')
    try:
        translation, ms = PIPELINE.translate_edited(text, language)
    except ValueError as error:
        raise gr.Error(str(error))
    return translation, _timing_table({'translate': ms})


def _model_line() -> str:
    # describe() returns two Markdown paragraphs; render them as one muted HTML line.
    text = html.escape(PIPELINE.describe().replace('\n\n', ' · '))
    parts = text.split('`')
    return ''.join(f'<code>{p}</code>' if i % 2 else p for i, p in enumerate(parts))


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
with gr.Blocks(title='Handwriting translator · CVDL Group 10') as demo:
    source_language = gr.State('')
    gr.HTML("""
<header class="mast">
  <div class="identity"><span class="seal" aria-hidden="true">文</span>
    <span>CVDL <span class="mast-slash">/</span> Group 10</span></div>
  <span class="pair">English <span class="arrow" aria-hidden="true">⇄</span> <span lang="zh-Hans">中文</span></span>
</header>
<section class="intro">
  <h1>Handwriting translator<span class="dot" aria-hidden="true">.</span></h1>
  <p>Read one handwritten line in English or Chinese and translate it into the other language.</p>
</section>""", padding=False)
    gr.HTML(f'<p class="model-line">{_model_line()}</p>', padding=False)

    with gr.Row(equal_height=False):
        with gr.Column():
            gr.HTML('<div class="panel-title"><h2>Source line</h2></div>'
                    '<p class="panel-note">Upload or paste a crop of one handwritten line. Webcam: hold one line close, '
                    'click the camera button to take the photo, then Read and translate.</p>', padding=False)
            # mirror=False: Gradio mirrors webcam photos by default, which reverses handwriting.
            # Prefer the rear camera on phones and keep Gradio's default 1920x1440 request.
            image = gr.Image(label='Handwritten line', type='pil', sources=['upload', 'webcam', 'clipboard'],
                             height=260, elem_classes='stage',
                             webcam_options=gr.WebcamOptions(mirror=False, constraints={'video': {
                                 'facingMode': 'environment', 'width': {'ideal': 1920}, 'height': {'ideal': 1440}}}))
            language = gr.Radio(list(LANGUAGE_CHOICES), value='Auto-detect', label='Source language')
            run = gr.Button('Read and translate  ↑', variant='primary', elem_classes='primary-action')
        with gr.Column():
            gr.HTML('<div class="panel-title"><h2>Reading</h2></div>'
                    '<p class="panel-note">The original OCR reading stays visible, so you can check it against the ink.</p>',
                    padding=False)
            detected = gr.Textbox(label='Language', interactive=False)
            transcription = gr.Textbox(label='Literal transcription (edit to correct, then re-translate)', lines=2)
            translation = gr.Textbox(label='Translation', lines=2, interactive=False)
            again = gr.Button('Translate edits  →', variant='secondary')
            timing = gr.Markdown(elem_classes='timing')

    gr.HTML('<div class="statusbar"><span class="dot" aria-hidden="true"></span>'
            '<p>Recognition: fine-tuned TrOCR (English) and PP-OCRv5 (Chinese). Translation: Marian (English→Chinese) '
            'and NLLB-200 distilled 600M (Chinese→English). Images are processed in memory and not stored.</p></div>',
            padding=False)

    run.click(read_and_translate, [image, language],
              [detected, transcription, translation, timing, source_language], concurrency_limit=1)
    again.click(retranslate, [transcription, source_language], [translation, timing], concurrency_limit=1)

if __name__ == '__main__':
    demo.queue(max_size=16).launch(theme=THEME, css=CSS, head=HEAD)
