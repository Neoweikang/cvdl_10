"""Model-free regression tests for OCR routing and translation orchestration."""
import unittest

from inference import run_line


class InferenceRoutingTests(unittest.TestCase):
    image = object()

    def setUp(self):
        self.recognition_calls = []
        self.translation_calls = []

    def timed(self, function, *args):
        return function(*args), 12.5

    def recognise(self, image, language):
        self.assertIs(image, self.image)
        self.recognition_calls.append(language)
        return self.recognised_text[language], 0.8

    def translate(self, text, language):
        self.translation_calls.append((text, language))
        return f'{language}:{text}'

    def run_case(self, language='auto', recognised=None):
        self.recognised_text = recognised or {'zh': '你好世界', 'en': 'hello world'}
        return run_line(self.image, language, self.timed, self.recognise, self.translate)

    def test_rejects_missing_image_and_unknown_language(self):
        with self.assertRaisesRegex(ValueError, 'Add an image'):
            run_line(None, 'auto', self.timed, self.recognise, self.translate)
        with self.assertRaisesRegex(ValueError, "language must be 'auto', 'en' or 'zh'"):
            self.run_case('fr')
        self.assertEqual(self.recognition_calls, [])

    def test_manual_english_uses_only_english_ocr(self):
        result = self.run_case('en')
        self.assertEqual(self.recognition_calls, ['en'])
        self.assertEqual(self.translation_calls, [('hello world', 'en')])
        self.assertEqual(result.routing, 'selected manually')
        self.assertEqual(result.language, 'en')

    def test_manual_chinese_uses_only_chinese_ocr(self):
        result = self.run_case('zh')
        self.assertEqual(self.recognition_calls, ['zh'])
        self.assertEqual(self.translation_calls, [('你好世界', 'zh')])
        self.assertEqual(result.language, 'zh')

    def test_auto_english_uses_chinese_detection_then_english_ocr(self):
        result = self.run_case('auto', {'zh': 'hello world', 'en': 'Hello world'})
        self.assertEqual(self.recognition_calls, ['zh', 'en'])
        self.assertEqual(self.translation_calls, [('Hello world', 'en')])
        self.assertEqual(result.routing, 'auto: script detection')
        self.assertEqual(result.language, 'en')

    def test_auto_chinese_uses_chinese_ocr_once(self):
        result = self.run_case('auto', {'zh': '你好世界', 'en': 'unused'})
        self.assertEqual(self.recognition_calls, ['zh'])
        self.assertEqual(result.routing, 'auto: script detection')
        self.assertEqual(result.language, 'zh')

    def test_mixed_script_routes_to_latin_majority(self):
        result = self.run_case('auto', {'zh': 'Hello 中', 'en': 'Hello 中'})
        self.assertEqual(self.recognition_calls, ['zh', 'en'])
        self.assertEqual(result.routing, 'auto: mixed script, routed by the majority')
        self.assertEqual(result.language, 'en')

    def test_mixed_script_tie_routes_to_chinese(self):
        result = self.run_case('auto', {'zh': 'A中', 'en': 'unused'})
        self.assertEqual(self.recognition_calls, ['zh'])
        self.assertEqual(result.routing, 'auto: mixed script, routed by the majority')
        self.assertEqual(result.language, 'zh')

    def test_empty_recognition_skips_translation(self):
        result = self.run_case('en', {'en': '', 'zh': 'unused'})
        self.assertEqual(self.translation_calls, [])
        self.assertEqual(result.translation, '')
        self.assertEqual(result.timings_ms['translate'], 0.0)

    def test_result_includes_stage_timings_and_total(self):
        result = self.run_case('zh')
        self.assertEqual(result.timings_ms, {'ocr_zh': 12.5, 'translate': 12.5})
        self.assertEqual(result.total_ms, 25.0)


if __name__ == '__main__':
    unittest.main()
