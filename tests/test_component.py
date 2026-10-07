"""Component tests: marker finding and marker-anchored hand search, on synthetic images."""
import numpy as np
import pytest

import retry_failed_marker as R
from conftest import HAND_TEMP_C, MARKER_OPTICAL, MARKER_TEMP_C, MARKER_THERMAL, mirror_x


def test_find_marker_candidates_locates_sticker(synthetic_scan):
    candidates = R.find_marker_candidates(
        synthetic_scan.celsius, synthetic_scan.optical, synthetic_scan.pip_info
    )
    assert candidates
    best = candidates[0]
    tx, ty, tr = MARKER_THERMAL
    assert best["thermal_xy"] == pytest.approx((tx, ty), abs=2)
    assert best["thermal_r"] == pytest.approx(tr, abs=2)
    assert best["optical_xy"] == pytest.approx(MARKER_OPTICAL[:2], abs=8)


def test_find_marker_candidates_empty_without_sticker(synthetic_scan):
    no_marker = synthetic_scan.celsius.copy()
    no_marker[no_marker == MARKER_TEMP_C] = HAND_TEMP_C
    assert R.find_marker_candidates(no_marker, synthetic_scan.optical, synthetic_scan.pip_info) == []


def test_detect_hand_with_marker_returns_validated_hand(
    synthetic_scan, marker, left_hand, fake_landmarker_factory
):
    landmarker = fake_landmarker_factory(left_hand)
    points, info = R.detect_hand_with_marker(
        landmarker, synthetic_scan.optical, marker, R.EXPECTED_CHIRALITY["LH"]
    )
    assert points == pytest.approx(left_hand, abs=1e-6)
    assert info["crop_half_size_r"] == R.CROP_HALF_SIZES[0]
    # Stops after the first zoom level once a hand validates
    assert landmarker.calls == len(R.CROP_ROTATIONS_DEG)


def test_detect_hand_with_marker_rejects_wrong_hand_side(
    synthetic_scan, marker, left_hand, fake_landmarker_factory
):
    landmarker = fake_landmarker_factory(mirror_x(left_hand, marker["optical_xy"][0]))
    points, info = R.detect_hand_with_marker(
        landmarker, synthetic_scan.optical, marker, R.EXPECTED_CHIRALITY["LH"]
    )
    assert points is None and info is None
