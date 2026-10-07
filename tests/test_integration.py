"""Integration tests: retry_scan end to end on one synthetic scan, with flyr and MediaPipe faked."""
import pytest

import retry_failed_marker as R
from conftest import HAND_TEMP_C, MARKER_TEMP_C, MARKER_THERMAL


@pytest.fixture
def unpack_returns(monkeypatch):
    def install(thermogram):
        monkeypatch.setattr(R.flyr, "unpack", lambda path: thermogram)
    return install


def test_retry_scan_recovers_temperatures(synthetic_scan, left_hand, unpack_returns, fake_landmarker_factory):
    unpack_returns(synthetic_scan)
    row = R.retry_scan(fake_landmarker_factory(left_hand), "scans/A001_CD_00_LH.jpg")

    assert row["Image_Name"] == "A001_CD_00_LH"
    assert row["Retry_Status"] == "RECOVERED"
    assert (row["Marker_Thermal_X"], row["Marker_Thermal_Y"]) == pytest.approx(MARKER_THERMAL[:2], abs=2)
    for segment in R.PHALANX_LANDMARKS:
        assert row[f"{segment}_Avg_C"] == pytest.approx(HAND_TEMP_C)


def test_retry_scan_rejects_hand_on_wrong_side(synthetic_scan, left_hand, unpack_returns, fake_landmarker_factory):
    unpack_returns(synthetic_scan)
    row = R.retry_scan(fake_landmarker_factory(left_hand), "scans/A001_CD_00_RH.jpg")
    assert row["Retry_Status"] == "NO HAND NEAR MARKER"
    assert "D3_Proximal_Avg_C" not in row


def test_retry_scan_reports_missing_marker(synthetic_scan, left_hand, unpack_returns, fake_landmarker_factory):
    synthetic_scan.celsius[synthetic_scan.celsius == MARKER_TEMP_C] = HAND_TEMP_C
    unpack_returns(synthetic_scan)
    row = R.retry_scan(fake_landmarker_factory(left_hand), "scans/A001_CD_00_LH.jpg")
    assert row["Retry_Status"] == "MARKER NOT FOUND"


def test_retry_scan_handles_unreadable_file(monkeypatch):
    def broken(path):
        raise ValueError("not a FLIR file")
    monkeypatch.setattr(R.flyr, "unpack", broken)
    assert R.retry_scan(None, "scans/bad.jpg")["Retry_Status"] == "UNPACK FAILED"
