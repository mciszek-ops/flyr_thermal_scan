"""Unit tests: pure geometry/scoring helpers, no images or models."""
from types import SimpleNamespace

import numpy as np
import pytest

import retry_failed_marker as R
from main_pipeline import optical_to_thermal
from conftest import make_left_hand, mirror_x


def test_thermal_to_optical_inverts_optical_to_thermal():
    pip = SimpleNamespace(crop_box=(10, 5, 310, 235), real_to_ir=1.3, offset_x=-25, offset_y=12)
    opt_w, opt_h, therm_w = 1280, 960, 320

    for u, v in [(0.5, 0.5), (0.2, 0.7), (0.8, 0.3)]:
        tx, ty = optical_to_thermal(u, v, opt_w, opt_h, therm_w, pip)
        ox, oy, _ = R.thermal_to_optical(tx, ty, opt_w, opt_h, therm_w, pip)
        assert ox == pytest.approx(u * opt_w)
        assert oy == pytest.approx(v * opt_h)


def test_crop_transform_puts_marker_at_crop_centre():
    m = R.marker_crop_transform((640.0, 480.0), half_size=100, angle_deg=30)
    centre = m @ np.array([640.0, 480.0, 1.0])
    assert centre == pytest.approx([R.CROP_OUTPUT_PX / 2] * 2)


def test_hand_chirality_left_vs_mirrored():
    hand = make_left_hand(640, 480, 48)
    assert R.hand_chirality(hand) == R.EXPECTED_CHIRALITY["LH"]
    assert R.hand_chirality(mirror_x(hand, 640)) == R.EXPECTED_CHIRALITY["RH"]


def test_marker_score_high_for_disc_zero_for_flat_image():
    flat = np.full((240, 320), 30.0, dtype=np.float32)
    assert R.score_marker_candidate(flat, 160, 120, 12) == 0.0

    disc = flat.copy()
    yy, xx = np.mgrid[:240, :320]
    disc[np.hypot(xx - 160, yy - 120) <= 12] = 25.0
    assert R.score_marker_candidate(disc, 160, 120, 12) > 1.0


class TestValidateAgainstMarker:
    def test_accepts_matching_hand(self, left_hand, marker):
        assert R.validate_against_marker(left_hand, marker, R.EXPECTED_CHIRALITY["LH"]) is not None

    def test_rejects_mirrored_finger_order(self, left_hand, marker):
        assert R.validate_against_marker(left_hand, marker, R.EXPECTED_CHIRALITY["RH"]) is None

    def test_rejects_hand_too_small_for_marker(self, marker):
        tiny = make_left_hand(*marker["optical_xy"], unit=marker["optical_r"] / 4)
        assert R.validate_against_marker(tiny, marker) is None

    def test_rejects_hand_away_from_marker(self, left_hand, marker):
        assert R.validate_against_marker(left_hand + [500, 0], marker) is None
