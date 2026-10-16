"""Reference cosine LR indexed only by successful actual Adam updates."""

import math


def cosine_learning_rate(update_index, total_updates, *, start=3e-5, end=3e-6):
    if (type(update_index) is not int or type(total_updates) is not int or total_updates < 2
            or not 0 <= update_index < total_updates
            or not all(math.isfinite(value) for value in (start, end)) or not 0 < end <= start):
        raise ValueError("Invalid successful-Adam cosine coordinate")
    if update_index == 0:
        return float(start)
    if update_index == total_updates - 1:
        return float(end)
    return float(end + 0.5 * (start - end) * (1 + math.cos(math.pi * update_index / (total_updates - 1))))
