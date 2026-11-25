"""Dependency-free loading and validation for labelled benchmark CSV rows."""
import csv
import sys
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    """Read benchmark rows and validate their required fields and image files."""
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)
    missing = {'image_path', 'text', 'language'} - set(headers)
    if missing:
        sys.exit(f'{path} is missing columns: {", ".join(sorted(missing))}')
    for row in rows:
        if row['language'] not in ('en', 'zh'):
            sys.exit(f'Language must be en or zh, got {row["language"]!r}')
        image = Path(row['image_path'])
        row['image_path'] = image if image.is_absolute() else path.parent / image
        if not row['image_path'].is_file():
            sys.exit(f'Image not found: {row["image_path"]}')
    return rows
