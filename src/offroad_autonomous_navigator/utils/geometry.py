import math


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi