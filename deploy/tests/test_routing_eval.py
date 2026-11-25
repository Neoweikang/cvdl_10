"""Tests for aggregate automatic language-routing evaluation."""
import unittest

from routing_eval import evaluate_routing


class RoutingEvaluationTests(unittest.TestCase):
    def test_reports_overall_per_language_and_confusion_counts(self):
        rows = [
            {'text': 'hello world', 'predicted': 'en'},
            {'text': 'hello 世界', 'predicted': 'zh'},
            {'text': '你好世界', 'predicted': 'zh'},
            {'text': 'A中', 'predicted': 'en'},
        ]

        result = evaluate_routing(rows, lambda row: row['predicted'])

        self.assertEqual(result['lines'], 4)
        self.assertEqual(result['correct'], 2)
        self.assertEqual(result['accuracy_percent'], 50.0)
        self.assertEqual(result['by_expected_language']['en'], {
            'lines': 2, 'correct': 1, 'accuracy_percent': 50.0,
        })
        self.assertEqual(result['by_expected_language']['zh'], {
            'lines': 2, 'correct': 1, 'accuracy_percent': 50.0,
        })
        self.assertEqual(result['confusion'], {
            'en': {'en': 1, 'zh': 1},
            'zh': {'en': 1, 'zh': 1},
        })

    def test_empty_input_has_no_accuracy_number(self):
        result = evaluate_routing([], lambda _row: 'en')

        self.assertEqual(result['lines'], 0)
        self.assertIsNone(result['accuracy_percent'])
        self.assertIsNone(result['by_expected_language']['en']['accuracy_percent'])

    def test_rejects_unexpected_prediction(self):
        with self.assertRaisesRegex(ValueError, 'must be en or zh'):
            evaluate_routing([{'text': 'hello'}], lambda _row: 'fr')


if __name__ == '__main__':
    unittest.main()

