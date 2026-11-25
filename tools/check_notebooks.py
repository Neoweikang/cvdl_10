"""Check notebook JSON and basic cell structure without running notebook code."""
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / 'notebooks'
CELL_TYPES = {'code', 'markdown', 'raw'}


def validate_notebook(path: Path) -> int:
    try:
        notebook = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'{path.relative_to(ROOT)}: cannot read valid UTF-8 notebook JSON: {error}') from error

    if not isinstance(notebook, dict) or notebook.get('nbformat') != 4:
        raise ValueError(f'{path.relative_to(ROOT)}: expected a Jupyter notebook in format 4')
    cells = notebook.get('cells')
    if not isinstance(cells, list) or not cells:
        raise ValueError(f'{path.relative_to(ROOT)}: expected a non-empty cells list')

    seen_ids = set()
    for index, cell in enumerate(cells):
        prefix = f'{path.relative_to(ROOT)} cell {index}'
        if not isinstance(cell, dict) or cell.get('cell_type') not in CELL_TYPES:
            raise ValueError(f'{prefix}: invalid cell type or cell object')
        source = cell.get('source')
        if isinstance(source, list):
            if not all(isinstance(line, str) for line in source):
                raise ValueError(f'{prefix}: source list must contain only strings')
        elif not isinstance(source, str):
            raise ValueError(f'{prefix}: source must be a string or a list of strings')

        cell_id = cell.get('id')
        if cell_id is not None:
            if not isinstance(cell_id, str) or not cell_id:
                raise ValueError(f'{prefix}: cell id must be a non-empty string')
            if cell_id in seen_ids:
                raise ValueError(f'{prefix}: duplicate cell id {cell_id!r}')
            seen_ids.add(cell_id)

        if cell['cell_type'] == 'code' and not isinstance(cell.get('outputs', []), list):
            raise ValueError(f'{prefix}: code cell outputs must be a list')

    return len(cells)


def main() -> int:
    paths = sorted(NOTEBOOKS.glob('*.ipynb'))
    if not paths:
        print(f'No notebooks found in {NOTEBOOKS}', file=sys.stderr)
        return 1
    try:
        for path in paths:
            count = validate_notebook(path)
            print(f'OK {path.relative_to(ROOT)} ({count} cells)')
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
