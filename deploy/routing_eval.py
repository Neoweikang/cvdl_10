"""Aggregate automatic language-routing quality without retaining predictions."""
from helpers import choose_language, script_counts


LANGUAGES = ('en', 'zh')


def evaluate_routing(rows, predict_language):
    """Compare predicted routes with the majority script in each reference.

    Mixed-script ties follow the deployed rule and count as Chinese. The callback
    receives a row and must return ``en`` or ``zh``; only aggregate counts return.
    """
    confusion = {expected: {predicted: 0 for predicted in LANGUAGES}
                 for expected in LANGUAGES}

    for row in rows:
        expected = choose_language(script_counts(row['text']))
        predicted = predict_language(row)
        if predicted not in LANGUAGES:
            raise ValueError(f"Predicted language must be en or zh, got {predicted!r}")
        confusion[expected][predicted] += 1

    by_expected = {}
    total = correct_total = 0
    for language in LANGUAGES:
        counts = confusion[language]
        lines = sum(counts.values())
        correct = counts[language]
        total += lines
        correct_total += correct
        by_expected[language] = {
            'lines': lines,
            'correct': correct,
            'accuracy_percent': round(100 * correct / lines, 2) if lines else None,
        }

    return {
        'lines': total,
        'correct': correct_total,
        'accuracy_percent': round(100 * correct_total / total, 2) if total else None,
        'by_expected_language': by_expected,
        'confusion': confusion,
    }

