"""Line-level handwriting OCR and translation for the deployed app.

This module reproduces the inference path of notebooks/Prototype_Cleaned.ipynb
without Colab or Google Drive:

  Cell 4   OCR loading and `transcribe`: greedy TrOCR for English, CTC decoding
           for PP-OCRv5 Chinese, lower-quintile token probability as confidence.
  Cell 24  Marian English -> Chinese with the >>cmn_Hans<< target tag.
  Cell 28  NLLB-200 (distilled 600M) Chinese -> English.
  Cell 29  Fixed routes: Marian for EN->ZH, NLLB for ZH->EN, 4 beams.
  Cell 47  Script-based language routing (Han vs Latin character counts).

Deployment adds model optimisation:
  int8  PyTorch dynamic quantisation of every nn.Linear layer (CPU only).
  fp16  Half-precision weights and activations (CUDA GPU only).
"""
from __future__ import annotations

import gc
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForSeq2SeqLM,
    AutoModelForTextRecognition,
    AutoTokenizer,
    MarianMTModel,
    MarianTokenizer,
    RobertaTokenizer,
    TrOCRProcessor,
    ViTImageProcessor,
    VisionEncoderDecoderModel,
)
from helpers import HAN_RANGES, choose_language, script_counts, summarise_ms

# Foundation models and the revisions recorded by the project notebook.
BASE_OCR = {
    'en': 'microsoft/trocr-base-handwritten',
    'zh': 'PaddlePaddle/PP-OCRv5_server_rec_safetensors',
}
BASE_OCR_REVISIONS = {
    'en': 'eaacaf452b06415df8f10bb6fad3a4c11e609406',
    'zh': '542979d7cc3791732bb12af35313a6840952d79f',
}
MARIAN_ID = 'Helsinki-NLP/opus-mt-en-zh'
MARIAN_REVISION = '408d9bc410a388e1d9aef112a2daba955b945255'
MARIAN_ZH_TAG = '>>cmn_Hans<<'  # Requests Simplified Chinese output.
NLLB_ID = 'facebook/nllb-200-distilled-600M'
NLLB_REVISION = 'f8d333a098d19b4fd9a8b18f94170487ad3f821d'
NLLB_CODES = {'en': 'eng_Latn', 'zh': 'zho_Hans'}

# Decoding settings copied from the notebook.
OCR_MAX_NEW_TOKENS = 128
MT_BEAMS = 4
MT_MAX_INPUT_TOKENS = 512
MT_MAX_NEW_TOKENS = 256

OPTIMISATIONS = ('none', 'fp16', 'int8')
LANGUAGE_NAMES = {'en': 'English', 'zh': 'Chinese'}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def resolve_optimisation(requested: str | None, device: str) -> str:
    """Map 'auto' to the right optimisation for the device and reject bad pairs."""
    requested = (requested or 'auto').strip().lower()
    if requested == 'auto':
        return 'fp16' if device == 'cuda' else 'int8'
    if requested not in OPTIMISATIONS:
        raise ValueError(f'OPTIMISE must be auto, {", ".join(OPTIMISATIONS)}; got {requested!r}.')
    if requested == 'int8' and device != 'cpu':
        raise ValueError('int8 dynamic quantisation runs on CPU only. Use fp16 on a GPU, or set DEVICE=cpu.')
    if requested == 'fp16' and device != 'cuda':
        raise ValueError('fp16 needs a CUDA GPU. Use int8 on CPU.')
    return requested


@dataclass
class Settings:
    en_ocr: str                 # Local folder or Hugging Face repo id.
    zh_ocr: str
    device: str = 'cpu'
    optimisation: str = 'none'
    optimise_ocr: bool = True   # False keeps OCR in FP32 and optimises translation only.

    @classmethod
    def from_env(cls) -> 'Settings':
        """Read EN_OCR, ZH_OCR, TRAINED_DIR, DEVICE, OPTIMISE and OPTIMISE_SCOPE."""
        device = (os.getenv('DEVICE') or ('cuda' if torch.cuda.is_available() else 'cpu')).lower()
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('DEVICE=cuda but PyTorch cannot see a GPU. Install the CUDA build of PyTorch.')
        trained = os.getenv('TRAINED_DIR')

        def pick(lang: str, variable: str) -> str:
            if os.getenv(variable):
                return os.environ[variable]
            if trained:
                folder = Path(trained).expanduser() / lang / 'weights'
                if not (folder / 'config.json').is_file():
                    raise FileNotFoundError(
                        f'{folder} has no config.json. TRAINED_DIR must be a saved training run '
                        'containing en/weights and zh/weights.')
                return str(folder)
            return BASE_OCR[lang]

        scope = (os.getenv('OPTIMISE_SCOPE') or 'all').strip().lower()
        if scope not in ('all', 'mt'):
            raise ValueError("OPTIMISE_SCOPE must be 'all' or 'mt'.")
        return cls(en_ocr=pick('en', 'EN_OCR'), zh_ocr=pick('zh', 'ZH_OCR'), device=device,
                   optimisation=resolve_optimisation(os.getenv('OPTIMISE'), device),
                   optimise_ocr=(scope == 'all'))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def optimise_model(model: torch.nn.Module, mode: str) -> torch.nn.Module:
    if mode == 'fp16':
        return model.half()
    if mode == 'int8':
        # Weights stored as int8; activations quantised on the fly at inference.
        return torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return model


def _tensor_bytes(obj, seen: set) -> int:
    if isinstance(obj, torch.Tensor):
        try:
            key = (obj.data_ptr(), obj.numel())
        except RuntimeError:
            key = (id(obj), obj.numel())
        if key in seen:  # Tied weights (e.g. shared embeddings) count once.
            return 0
        seen.add(key)
        return obj.numel() * obj.element_size()
    if isinstance(obj, (tuple, list)):
        return sum(_tensor_bytes(item, seen) for item in obj)
    return 0


def model_size_mb(model: torch.nn.Module) -> float:
    """In-memory size of the weights, including int8 packed parameters."""
    seen: set = set()
    return sum(_tensor_bytes(value, seen) for value in model.state_dict().values()) / 1e6


def float_dtype(model: torch.nn.Module) -> torch.dtype:
    for parameter in model.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _short(source: str) -> str:
    path = Path(source)
    return f'{path.parent.parent.name}/{path.parent.name}/{path.name}' if path.exists() else source


@dataclass
class LineResult:
    language: str               # 'en' or 'zh'
    routing: str                # How the language was chosen.
    text: str
    confidence: float | None    # Lower-quintile token probability (0-1).
    translation: str
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.models: dict[str, tuple[torch.nn.Module, object]] = {}
        self.sizes_mb: dict[str, float] = {}
        self.load_seconds: float | None = None
        self._lock = threading.Lock()

    # ----- loading -------------------------------------------------------- #
    def _revision(self, source: str, lang: str) -> dict:
        return {'revision': BASE_OCR_REVISIONS[lang]} if source == BASE_OCR[lang] else {}

    def _en_tokenizer_file(self, source: str, name: str) -> str:
        local = Path(source) / name
        if local.is_file():
            return str(local)
        if source != BASE_OCR['en'] and not Path(source).exists():
            try:  # A Hub copy of the trained model may include its tokenizer files.
                return hf_hub_download(source, name)
            except Exception:
                pass
        # The tokenizer was not fine-tuned, so the foundation files are equivalent.
        return hf_hub_download(BASE_OCR['en'], name, revision=BASE_OCR_REVISIONS['en'])

    def _load_en_ocr(self):
        source = self.settings.en_ocr
        revision = self._revision(source, 'en')
        tokenizer = RobertaTokenizer(vocab=self._en_tokenizer_file(source, 'vocab.json'),
                                     merges=self._en_tokenizer_file(source, 'merges.txt'),
                                     model_max_length=512)
        processor = TrOCRProcessor(ViTImageProcessor.from_pretrained(source, **revision), tokenizer)
        model = VisionEncoderDecoderModel.from_pretrained(source, dtype=torch.float32, **revision)
        model.config.decoder_start_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        return model, processor

    def _load_zh_ocr(self):
        source = self.settings.zh_ocr
        revision = self._revision(source, 'zh')
        processor = AutoImageProcessor.from_pretrained(source, **revision)
        model = AutoModelForTextRecognition.from_pretrained(source, dtype=torch.float32, **revision)
        return model, processor

    def _load_marian(self):
        tokenizer = MarianTokenizer.from_pretrained(MARIAN_ID, revision=MARIAN_REVISION)
        if MARIAN_ZH_TAG not in tokenizer.get_vocab():
            raise RuntimeError('The Marian tokenizer is missing the Simplified Chinese tag.')
        model = MarianMTModel.from_pretrained(MARIAN_ID, revision=MARIAN_REVISION, dtype=torch.float32)
        return model, tokenizer

    def _load_nllb(self):
        tokenizer = AutoTokenizer.from_pretrained(NLLB_ID, revision=NLLB_REVISION,
                                                  src_lang=NLLB_CODES['zh'], legacy_behaviour=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_ID, revision=NLLB_REVISION, dtype=torch.float32)
        return model, tokenizer

    def _register(self, name: str, loaded, optimise: bool):
        model, helper = loaded
        model.eval()
        mode = self.settings.optimisation if optimise else 'none'
        model = optimise_model(model, mode).to(self.settings.device).eval()
        self.models[name] = (model, helper)
        self.sizes_mb[name] = model_size_mb(model)
        gc.collect()

    def load(self, warmup: bool = True) -> 'Pipeline':
        started = time.perf_counter()
        ocr = self.settings.optimise_ocr
        self._register('ocr_en', self._load_en_ocr(), ocr)
        self._register('ocr_zh', self._load_zh_ocr(), ocr)
        self._register('mt_en', self._load_marian(), True)   # English -> Chinese
        self._register('mt_zh', self._load_nllb(), True)     # Chinese -> English
        if warmup:  # Blank-image smoke test, as in the notebook restore cell.
            blank = Image.new('RGB', (256, 64), 'white')
            for lang in ('en', 'zh'):
                text, _ = self.recognise(blank, lang)
                if not isinstance(text, str):
                    raise TypeError('OCR warm-up returned non-text.')
            self.translate('Hello.', 'en')
            self.translate('你好。', 'zh')
        self._sync()
        self.load_seconds = time.perf_counter() - started
        return self

    def unload(self):
        self.models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- inference ------------------------------------------------------ #
    def _sync(self):
        if self.settings.device == 'cuda':
            torch.cuda.synchronize()

    def _timed(self, function, *args):
        self._sync()
        started = time.perf_counter()
        result = function(*args)
        self._sync()
        return result, (time.perf_counter() - started) * 1000

    @torch.inference_mode()
    def recognise(self, image: Image.Image, lang: str) -> tuple[str, float | None]:
        """Read one cropped line. Returns text and lower-quintile token probability."""
        model, processor = self.models[f'ocr_{lang}']
        inputs = processor(images=image.convert('RGB'), return_tensors='pt')
        pixels = inputs['pixel_values'].to(self.settings.device, dtype=float_dtype(model))
        if lang == 'en':
            tokenizer = processor.tokenizer
            output = model.generate(
                pixels, max_new_tokens=OCR_MAX_NEW_TOKENS, do_sample=False, num_beams=1,
                return_dict_in_generate=True, output_scores=True,
                decoder_start_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            sequence = output.sequences
            text = processor.batch_decode(sequence, skip_special_tokens=True)[0].strip()
            ids = sequence[0, 1:]  # Skip the decoder start token.
            chosen = [step[0].float().softmax(-1)[token] for step, token in zip(output.scores, ids)]
            mask = torch.ones_like(ids, dtype=torch.bool)
            for special in tokenizer.all_special_ids:
                mask &= ids != special
            probabilities = torch.stack(chosen)[mask[:len(chosen)]] if chosen else torch.empty(0)
        else:
            output = model(pixel_values=pixels)
            # The head already applies softmax. Collapse repeats and blanks (CTC).
            probability, ids = output.last_hidden_state[0].float().max(-1)
            mask = ids != 0
            mask[1:] = mask[1:] & (ids[1:] != ids[:-1])
            text = ''.join(processor.character_list[i] for i in ids[mask].tolist()).strip()
            probabilities = probability[mask]
        confidence = float(torch.quantile(probabilities.float(), .2)) if probabilities.numel() else None
        return text, confidence

    @torch.inference_mode()
    def translate(self, text: str, source_lang: str) -> str:
        if source_lang not in ('en', 'zh'):
            raise ValueError("source_lang must be 'en' or 'zh'.")
        text = text.strip()
        if not text:
            return ''
        model, tokenizer = self.models[f'mt_{source_lang}']
        if source_lang == 'en':
            prepared, extra = f'{MARIAN_ZH_TAG} {text}', {}
        else:
            tokenizer.src_lang = NLLB_CODES['zh']
            prepared = text
            extra = {'forced_bos_token_id': tokenizer.convert_tokens_to_ids(NLLB_CODES['en'])}
        inputs = tokenizer(prepared, return_tensors='pt', truncation=False)
        tokens = inputs['input_ids'].shape[1]
        limit = min(MT_MAX_INPUT_TOKENS, model.config.max_position_embeddings)
        if tokens > limit:
            raise ValueError(f'Input has {tokens} tokens; the limit is {limit}. Use a shorter line.')
        generated = model.generate(
            **inputs.to(self.settings.device), **extra, max_length=None,
            max_new_tokens=MT_MAX_NEW_TOKENS, num_beams=MT_BEAMS, num_return_sequences=1,
            do_sample=False, early_stopping=True, length_penalty=1.0, use_cache=True)
        return tokenizer.decode(generated[0], skip_special_tokens=True).strip()

    def read_line(self, image: Image.Image | None, language: str = 'auto') -> LineResult:
        """OCR one line, choose the language if needed, then translate."""
        if image is None:
            raise ValueError('Add an image of one handwritten line.')
        if language not in ('auto', 'en', 'zh'):
            raise ValueError("language must be 'auto', 'en' or 'zh'.")
        with self._lock:
            timings: dict[str, float] = {}
            if language in ('en', 'zh'):
                (text, confidence), timings[f'ocr_{language}'] = self._timed(self.recognise, image, language)
                routing = 'selected manually'
            else:
                # PP-OCRv5 reads both scripts, so it decides the language first.
                (text, confidence), timings['ocr_zh'] = self._timed(self.recognise, image, 'zh')
                counts = script_counts(text)
                language = choose_language(counts)
                if language == 'en':
                    # Mostly Latin: TrOCR is the stronger English reader.
                    (text, confidence), timings['ocr_en'] = self._timed(self.recognise, image, 'en')
                mixed = counts['en'] and counts['zh']
                routing = ('auto: mixed script, routed by the majority' if mixed
                           else 'auto: script detection')
            translation, timings['translate'] = self._timed(self.translate, text, language) if text else ('', 0.0)
            return LineResult(language, routing, text, confidence, translation, timings)

    def translate_edited(self, text: str, language: str) -> tuple[str, float]:
        with self._lock:
            return self._timed(self.translate, text, language)

    # ----- reporting ------------------------------------------------------ #
    def describe(self) -> str:
        s = self.settings
        mode = {'none': 'FP32 (not optimised)', 'fp16': 'FP16 half precision',
                'int8': 'int8 dynamic quantisation'}[s.optimisation]
        if s.optimisation != 'none' and not s.optimise_ocr:
            mode += ' (translation only; OCR kept in FP32)'
        total = sum(self.sizes_mb.values())
        load = f'{self.load_seconds:.0f} s' if self.load_seconds is not None else 'not loaded'
        return (f'Device: {s.device.upper()} · Models: {mode} · Weights in memory: {total:,.0f} MB · '
                f'Start-up: {load}\n\nEnglish OCR: `{_short(s.en_ocr)}` · Chinese OCR: `{_short(s.zh_ocr)}`')


