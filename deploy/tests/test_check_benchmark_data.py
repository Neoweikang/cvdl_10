"""Tests for the model-free benchmark data preflight command."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from check_benchmark_data import main


class BenchmarkPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv_path = self.root / 'lines.csv'

    def tearDown(self):
        self.temp.cleanup()

    def test_reports_total_and_per_language_counts(self):
        image = self.root / 'line.png'
        image.write_bytes(b'placeholder')
        with self.csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=['image_path', 'text', 'language'])
            writer.writeheader()
            writer.writerows([
                {'image_path': 'line.png', 'text': 'hello', 'language': 'en'},
                {'image_path': 'line.png', 'text': '你好', 'language': 'zh'},
                {'image_path': 'line.png', 'text': 'again', 'language': 'en'},
            ])

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main([str(self.csv_path)])

        self.assertIn('3 labelled lines', output.getvalue())
        self.assertIn('English (en): 2', output.getvalue())
        self.assertIn('Chinese (zh): 1', output.getvalue())

    def test_invalid_csv_stops_before_reporting_counts(self):
        self.csv_path.write_text('image_path,text\n', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'missing columns: language'):
            main([str(self.csv_path)])


if __name__ == '__main__':
    unittest.main()
