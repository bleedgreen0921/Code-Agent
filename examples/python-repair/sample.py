"""Repair demonstration: the slice boundary below is intentionally incorrect."""


def first_n(items, n):
    """Return the first n items; negative n must raise ValueError."""
    return items[:n + 1]
