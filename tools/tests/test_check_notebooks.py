"""Unit tests for notebook structure validation; no notebook code is executed."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import check_notebooks


class ValidateNotebookTests(unittest.TestCase):
    def validate(self, notebook):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'sample.ipynb'
            with patch.object(check_notebooks, 'ROOT', root):
                if isinstance(notebook, str):
                    path.write_text(notebook, encoding='utf-8')
                else:
                    path.write_text(json.dumps(notebook), encoding='utf-8')
                return check_notebooks.validate_notebook(path)

    def test_accepts_valid_notebook(self):
        notebook = {
            'nbformat': 4,
            'cells': [{'cell_type': 'markdown', 'id': 'intro', 'source': ['# Demo']}],
        }
        self.assertEqual(self.validate(notebook), 1)

    def test_rejects_malformed_json(self):
        with self.assertRaisesRegex(ValueError, 'cannot read valid UTF-8 notebook JSON'):
            self.validate('{not json')

    def test_rejects_duplicate_cell_ids(self):
        notebook = {
            'nbformat': 4,
            'cells': [
                {'cell_type': 'markdown', 'id': 'same', 'source': []},
                {'cell_type': 'code', 'id': 'same', 'source': [], 'outputs': []},
            ],
        }
        with self.assertRaisesRegex(ValueError, 'duplicate cell id'):
            self.validate(notebook)

    def test_rejects_non_string_cell_source(self):
        notebook = {
            'nbformat': 4,
            'cells': [{'cell_type': 'markdown', 'source': [123]}],
        }
        with self.assertRaisesRegex(ValueError, 'source list must contain only strings'):
            self.validate(notebook)

    def test_rejects_non_list_code_outputs(self):
        notebook = {
            'nbformat': 4,
            'cells': [{'cell_type': 'code', 'source': '', 'outputs': {}}],
        }
        with self.assertRaisesRegex(ValueError, 'outputs must be a list'):
            self.validate(notebook)


if __name__ == '__main__':
    unittest.main()
