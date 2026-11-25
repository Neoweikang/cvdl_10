"""Find broken relative file links in Markdown without network access."""
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r'!?\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))')
FENCE = re.compile(r'^\s*(`{3,}|~{3,})')


def check_markdown_file(markdown_path: Path) -> list[str]:
    """Return line-numbered errors for missing relative targets in one Markdown file."""
    try:
        lines = markdown_path.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeError) as error:
        return [f'{markdown_path}: cannot read UTF-8 Markdown: {error}']

    errors = []
    fence_char = None
    fence_length = 0
    for line_number, line in enumerate(lines, 1):
        fence_match = FENCE.match(line)
        if fence_match:
            marker = fence_match.group(1)
            if fence_char is None:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = None, 0
            continue
        if fence_char is not None:
            continue

        for match in LINK.finditer(line):
            target_text = (match.group(1) or match.group(2)).strip()
            target_url = urlsplit(target_text)
            if target_url.scheme or target_text.startswith('//') or not target_url.path:
                continue

            target_path = Path(unquote(target_url.path))
            if not target_path.is_absolute():
                target_path = markdown_path.parent / target_path
            if not target_path.exists():
                display_path = markdown_path
                try:
                    display_path = markdown_path.relative_to(ROOT)
                except ValueError:
                    pass
                errors.append(
                    f'{display_path}:{line_number}: missing local link target {target_text!r}'
                )
    return errors


def main() -> int:
    markdown_files = sorted(path for path in ROOT.rglob('*.md')
                            if not any(part.startswith('.') for part in path.relative_to(ROOT).parts))
    if not markdown_files:
        print('No Markdown files found.', file=sys.stderr)
        return 1

    errors = [error for path in markdown_files for error in check_markdown_file(path)]
    if errors:
        print('\n'.join(errors), file=sys.stderr)
        return 1
    print(f'OK: checked local links in {len(markdown_files)} Markdown files.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
