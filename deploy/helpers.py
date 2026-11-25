"""Dependency-free helpers shared by the deployment pipeline and benchmark."""
from __future__ import annotations

import statistics
import unicodedata


HAN_RANGES = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
              (0x20000, 0x2EBEF), (0x30000, 0x323AF))


def script_counts(text: str) -> dict[str, int]:
    """Count Han, Latin and other letters, after compatibility normalisation."""
    counts = {'zh': 0, 'en': 0, 'other': 0}
    for char in unicodedata.normalize('NFKC', text):
        code = ord(char)
        if any(low <= code <= high for low, high in HAN_RANGES):
            counts['zh'] += 1
        elif char.isalpha():
            counts['en' if 'LATIN' in unicodedata.name(char, '') else 'other'] += 1
    return counts


def choose_language(counts: dict[str, int]) -> str:
    """Choose English for a Latin majority; choose Chinese for ties or Han majority."""
    return 'en' if counts.get('en', 0) > counts.get('zh', 0) else 'zh'


def edit_distance(a: str, b: str) -> int:
    """Return character-level Levenshtein distance using O(len(b)) memory."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def summarise_ms(values: list[float]) -> tuple[float, float]:
    """Return median and 95th percentile milliseconds (NaNs for no samples)."""
    if not values:
        return float('nan'), float('nan')
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
    return statistics.median(ordered), p95
