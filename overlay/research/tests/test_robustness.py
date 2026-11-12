"""Heading-sensor robustness helpers."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research import robustness as RB  # noqa: E402
from research.common import HEADING_NOISE_DEFAULTS  # noqa: E402


def test_noise_overrides():
    o = RB.noise_overrides(5)
    assert len(o) == len(HEADING_NOISE_DEFAULTS)
    kv = dict(x.split('=') for x in o)
    assert math.isclose(float(kv[f'{RB.SENSOR}.bias_std']), 0.01)
    assert all(float(v) == 0 for v in dict(x.split('=') for x in RB.noise_overrides(0)).values())


def test_expected_sigma():
    assert RB.expected_heading_sigma_deg(0, 12.4) == 0
    s1 = RB.expected_heading_sigma_deg(1, 12.4)
    assert 1.0 < s1 < 2.5  # ~1.7 deg at the end of a 12.4 s skateboard episode
    assert math.isclose(RB.expected_heading_sigma_deg(10, 12.4), 10 * s1, rel_tol=1e-9)
