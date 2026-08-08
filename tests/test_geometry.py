import math

import pytest

from offroad_autonomous_navigator.utils.geometry import normalize_angle

# Test cases for the normalize_angle function

def test_normalize_angle_already_in_range() -> None:
    """Verifies that angles within the range [-pi, pi] are not modified."""
    assert normalize_angle(0.0) == pytest.approx(0.0)
    assert normalize_angle(math.pi / 2) == pytest.approx(math.pi / 2)
    assert normalize_angle(-math.pi / 2) == pytest.approx(-math.pi / 2)


def test_normalize_angle_slightly_exceeding_positive_pi() -> None:
    """An angle that slightly exceeds pi should wrap around to -pi."""
    angle = math.pi + 0.1
    expected = -math.pi + 0.1
    assert normalize_angle(angle) == pytest.approx(expected)


def test_normalize_angle_slightly_exceeding_negative_pi() -> None:
    """An angle that slightly exceeds -pi should wrap around to pi."""
    angle = -math.pi - 0.1
    expected = math.pi - 0.1
    assert normalize_angle(angle) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("input_angle", "expected_angle"),
    [
        (2 * math.pi, 0.0),                     # 1 full positive turn (360º)
        (-2 * math.pi, 0.0),                    # 1 full negative turn (-360º)
        (3 * math.pi, -math.pi),                # 1.5 turns (3pi -> -pi using the % modulus)
        (10.5 * math.pi, 0.5 * math.pi),        # Multiple positive turns (5 turns + pi/2)
        (-7.25 * math.pi, 0.75 * math.pi),      # Multiple negative turns
    ],
)
def test_normalize_angle_multiple_turns(input_angle: float, expected_angle: float) -> None:
    """Verifies that the % modulus handles multiple turns and large values correctly."""
    assert normalize_angle(input_angle) == pytest.approx(expected_angle)