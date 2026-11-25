"""Tests for the offline local Markdown link checker."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import check_markdown_links


class CheckMarkdownFileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.docs = self.root / 'docs'
        self.docs.mkdir()
        self.markdown = self.docs / 'guide.md'
        self.root_file = self.root / 'README.md'
        self.root_file.write_text('# Project\n', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def check(self, content):
        self.markdown.write_text(content, encoding='utf-8')
        with patch.object(check_markdown_links, 'ROOT', self.root):
            return check_markdown_links.check_markdown_file(self.markdown)

    def test_accepts_existing_relative_file_and_directory_links(self):
        (self.docs / 'setup details.md').write_text('# Setup\n', encoding='utf-8')
        (self.root / 'assets').mkdir()
        errors = self.check(
            '[Home](../README.md)\n[Setup](<setup%20details.md> "guide")\n[Assets](../assets/)\n'
        )
        self.assertEqual(errors, [])

    def test_reports_missing_local_target_with_line_number(self):
        errors = self.check('First line\n[Missing](not-here.md)\n')
        self.assertEqual(len(errors), 1)
        self.assertIn('guide.md:2', errors[0])
        self.assertIn('not-here.md', errors[0])

    def test_ignores_external_and_fragment_only_links(self):
        errors = self.check('[Web](https://example.org/page) [Section](#intro)\n')
        self.assertEqual(errors, [])

    def test_ignores_link_examples_inside_fenced_code(self):
        errors = self.check('```markdown\n[Example](missing.md)\n```\n')
        self.assertEqual(errors, [])

    def test_checks_image_links_too(self):
        errors = self.check('![Diagram](missing-image.png)\n')
        self.assertEqual(len(errors), 1)
        self.assertIn('missing-image.png', errors[0])


if __name__ == '__main__':
    unittest.main()
