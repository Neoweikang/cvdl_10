"""Tests for benchmark CSV validation; no OCR models are loaded."""
import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_data import load_rows


class LoadRowsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.csv_path = self.root / 'lines.csv'

    def tearDown(self):
        self.temp.cleanup()

    def write_csv(self, headers, rows=()):
        with self.csv_path.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=headers)
            writer.writeheader()
            writer.writerows(rows)

    def test_loads_valid_row_and_resolves_relative_image_path(self):
        image_path = self.root / 'line.png'
        image_path.write_bytes(b'placeholder image bytes')
        self.write_csv(
            ['image_path', 'text', 'language'],
            [{'image_path': 'line.png', 'text': 'hello', 'language': 'en'}],
        )

        rows = load_rows(self.csv_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['image_path'], image_path)
        self.assertEqual(rows[0]['text'], 'hello')

    def test_rejects_missing_required_column(self):
        self.write_csv(['image_path', 'text'])
        with self.assertRaisesRegex(SystemExit, 'missing columns: language'):
            load_rows(self.csv_path)

    def test_rejects_invalid_language(self):
        self.write_csv(
            ['image_path', 'text', 'language'],
            [{'image_path': 'line.png', 'text': 'hello', 'language': 'fr'}],
        )
        with self.assertRaisesRegex(SystemExit, 'Language must be en or zh'):
            load_rows(self.csv_path)

    def test_rejects_missing_image_file(self):
        self.write_csv(
            ['image_path', 'text', 'language'],
            [{'image_path': 'missing.png', 'text': 'hello', 'language': 'en'}],
        )
        with self.assertRaisesRegex(SystemExit, 'Image not found'):
            load_rows(self.csv_path)

    def test_rejects_empty_csv_with_missing_column_message(self):
        self.csv_path.write_text('', encoding='utf-8')
        with self.assertRaisesRegex(SystemExit, 'missing columns'):
            load_rows(self.csv_path)


if __name__ == '__main__':
    unittest.main()
