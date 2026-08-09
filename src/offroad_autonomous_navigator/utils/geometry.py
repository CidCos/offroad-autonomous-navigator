import math


def normalize_angle(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the Euclidean distance between two points (x1, y1) and (x2, y2)."""
    return math.hypot(x2 - x1, y2 - y1)