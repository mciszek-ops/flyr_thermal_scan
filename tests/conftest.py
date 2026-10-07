import os
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

# main_pipeline loads hand_landmarker.task relative to the working directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)

# Synthetic scene: the optical image is exactly 4x the thermal grid (no parallax),
# with the marker in the centre of both.
THERM_W, THERM_H = 320, 240
OPT_W, OPT_H = 1280, 960
MARKER_THERMAL = (160, 120, 12)  # x, y, radius
MARKER_OPTICAL = (640.0, 480.0, 48.0)
HAND_TEMP_C = 30.0
MARKER_TEMP_C = 25.0
BACKGROUND_TEMP_C = 20.0


@pytest.fixture
def simple_pip():
    return SimpleNamespace(crop_box=(0, 0, THERM_W, THERM_H), real_to_ir=1.0, offset_x=0, offset_y=0)


def make_left_hand(cx, cy, unit):
    """
    21 MediaPipe-style landmarks (optical pixels) for a left hand seen from the
    back, fingers pointing up: pinky on the left, thumb on the right. `unit` is
    the marker radius; knuckle width is 6 units.
    """
    pts = np.zeros((21, 2))
    pts[0] = (cx, cy + 1.5 * unit)  # wrist
    for i, (dx, dy) in enumerate([(1.5, 1.0), (2.5, 0.2), (3.2, -0.6), (3.8, -1.4)], start=1):
        pts[i] = (cx + dx * unit, cy + dy * unit)  # thumb
    for mcp, dx in zip((5, 9, 13, 17), (3, 1, -1, -3)):  # index, middle, ring, pinky
        for j in range(4):
            pts[mcp + j] = (cx + dx * unit, cy - (1 + 1.5 * j) * unit)
    return pts


def mirror_x(pts, cx):
    out = pts.copy()
    out[:, 0] = 2 * cx - out[:, 0]
    return out


@pytest.fixture
def left_hand():
    x, y, r = MARKER_OPTICAL
    return make_left_hand(x, y, r)


@pytest.fixture
def marker():
    x, y, r = MARKER_OPTICAL
    return {"optical_xy": (x, y), "optical_r": r}


@pytest.fixture
def synthetic_scan(simple_pip):
    """A thermogram-like object containing one marker sticker on a warm hand."""
    celsius = np.full((THERM_H, THERM_W), HAND_TEMP_C, dtype=np.float32)
    celsius[200:] = BACKGROUND_TEMP_C  # cool background below the hand, as in real scans
    tx, ty, tr = MARKER_THERMAL
    cv2.circle(celsius, (tx, ty), tr, MARKER_TEMP_C, -1)

    optical = np.zeros((OPT_H, OPT_W, 3), dtype=np.uint8)
    optical[:] = (180, 120, 100)  # skin
    ox, oy, orad = MARKER_OPTICAL
    cv2.circle(optical, (int(ox), int(oy)), int(orad), (255, 255, 255), -1)

    return SimpleNamespace(celsius=celsius, optical=optical, pip_info=simple_pip)


class FakeLandmarker:
    """
    Stand-in for MediaPipe's HandLandmarker. Returns the given hand (optical
    pixels) projected into whatever crop it is shown. The crop transform is
    supplied by `transform_source`, a callable returning the latest affine matrix.
    """

    def __init__(self, hand_px, transform_source):
        self.hand_px = hand_px
        self.transform_source = transform_source
        self.calls = 0

    def detect(self, mp_image):
        self.calls += 1
        m = self.transform_source()
        size = mp_image.width
        crop_px = self.hand_px @ m[:, :2].T + m[:, 2]
        landmarks = [SimpleNamespace(x=x / size, y=y / size) for x, y in crop_px]
        return SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[[SimpleNamespace(score=0.9)]],
        )


@pytest.fixture
def fake_landmarker_factory(monkeypatch):
    """Build a FakeLandmarker that sees the crops retry_failed_marker makes."""
    import retry_failed_marker as R

    last = {}
    real_transform = R.marker_crop_transform

    def recording_transform(*args, **kwargs):
        last["m"] = real_transform(*args, **kwargs)
        return last["m"]

    monkeypatch.setattr(R, "marker_crop_transform", recording_transform)
    return lambda hand_px: FakeLandmarker(hand_px, lambda: last["m"])
