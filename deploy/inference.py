"""Model-free orchestration for one line of OCR and translation."""
from dataclasses import dataclass, field

from helpers import choose_language, script_counts


@dataclass
class LineResult:
    language: str
    routing: str
    text: str
    confidence: float | None
    translation: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


def run_line(image, language, timed, recognise, translate) -> LineResult:
    """Route a line to OCR and translation callbacks, recording each stage."""
    if image is None:
        raise ValueError('Add an image of one handwritten line.')
    if language not in ('auto', 'en', 'zh'):
        raise ValueError("language must be 'auto', 'en' or 'zh'.")

    timings: dict[str, float] = {}
    if language in ('en', 'zh'):
        (text, confidence), timings[f'ocr_{language}'] = timed(recognise, image, language)
        routing = 'selected manually'
    else:
        # PP-OCRv5 reads both scripts, so it decides the language first.
        (text, confidence), timings['ocr_zh'] = timed(recognise, image, 'zh')
        counts = script_counts(text)
        language = choose_language(counts)
        if language == 'en':
            # Mostly Latin: TrOCR is the stronger English reader.
            (text, confidence), timings['ocr_en'] = timed(recognise, image, 'en')
        mixed = counts['en'] and counts['zh']
        routing = ('auto: mixed script, routed by the majority' if mixed
                   else 'auto: script detection')

    if text:
        translation, timings['translate'] = timed(translate, text, language)
    else:
        translation, timings['translate'] = '', 0.0
    return LineResult(language, routing, text, confidence, translation, timings)
