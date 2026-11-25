"""Small regression tests for helpers that do not load any models."""
import unittest

from helpers import choose_language, edit_distance, script_counts, summarise_ms


class ScriptCountsTests(unittest.TestCase):
    def test_counts_english_chinese_and_other_letters(self):
        self.assertEqual(
            script_counts("Hello 世界 é"),
            {"zh": 2, "en": 6, "other": 0},
        )

    def test_normalises_full_width_latin(self):
        self.assertEqual(script_counts("ＡＢＣ"), {"zh": 0, "en": 3, "other": 0})

    def test_mixed_scripts_and_non_latin_letters(self):
        self.assertEqual(script_counts("A中Ж"), {"zh": 1, "en": 1, "other": 1})

    def test_ignores_digits_and_punctuation(self):
        self.assertEqual(script_counts("123!?，"), {"zh": 0, "en": 0, "other": 0})

    def test_empty_text(self):
        self.assertEqual(script_counts(""), {"zh": 0, "en": 0, "other": 0})


class LanguageRoutingTests(unittest.TestCase):
    def test_english_text_routes_to_english(self):
        self.assertEqual(choose_language(script_counts("hello world")), "en")

    def test_chinese_text_routes_to_chinese(self):
        self.assertEqual(choose_language(script_counts("你好世界")), "zh")

    def test_mixed_text_routes_to_latin_majority(self):
        self.assertEqual(choose_language(script_counts("hello 中")), "en")

    def test_mixed_text_routes_to_han_majority(self):
        self.assertEqual(choose_language(script_counts("hi 中文字")), "zh")

    def test_tie_and_empty_counts_default_to_chinese(self):
        self.assertEqual(choose_language(script_counts("A中")), "zh")
        self.assertEqual(choose_language({}), "zh")


class EditDistanceTests(unittest.TestCase):
    def test_identical_empty_and_substitution(self):
        self.assertEqual(edit_distance("", ""), 0)
        self.assertEqual(edit_distance("hand", "hand"), 0)
        self.assertEqual(edit_distance("cat", "cut"), 1)

    def test_insertions_and_deletions(self):
        self.assertEqual(edit_distance("cat", "cart"), 1)
        self.assertEqual(edit_distance("cart", "cat"), 1)

    def test_completely_different_and_unicode_text(self):
        self.assertEqual(edit_distance("abc", "xyz"), 3)
        self.assertEqual(edit_distance("你好", "你们好"), 1)


class SummariseMsTests(unittest.TestCase):
    def test_returns_median_and_p95(self):
        self.assertEqual(summarise_ms([10, 20, 30, 40, 50]), (30, 50))

    def test_even_sample_median_and_unsorted_input(self):
        self.assertEqual(summarise_ms([40, 10, 30, 20]), (25, 40))

    def test_single_sample(self):
        self.assertEqual(summarise_ms([12.5]), (12.5, 12.5))

    def test_empty_input_returns_nan_values(self):
        median, p95 = summarise_ms([])
        self.assertTrue(median != median)  # NaN is unequal to itself.
        self.assertTrue(p95 != p95)


if __name__ == "__main__":
    unittest.main()
