"""
End-to-end test: run the real script (real flyr + MediaPipe) on a real scan that
the default pipeline missed. Skipped when the scan data is not available, since
raw_thermal_scans/ is not in git.
"""
import os
import sys

import pandas as pd
import pytest

import retry_failed_marker as R

KNOWN_RECOVERABLE = "A071_CD_10_LH"
SCAN_PATH = os.path.join(R.THERMAL_INPUT_DIR, "LHand", f"{KNOWN_RECOVERABLE}.jpg")

pytestmark = pytest.mark.skipif(not os.path.exists(SCAN_PATH), reason=f"needs {SCAN_PATH}")


def test_main_recovers_failed_scan_and_merges_summary(tmp_path, monkeypatch):
    temp_cols = [f"{segment}_Avg_C" for segment in R.PHALANX_LANDMARKS]
    summary = pd.DataFrame([
        {"Image_Name": "ALREADY_OK_LH", **{c: 31.0 for c in temp_cols}},
        {"Image_Name": KNOWN_RECOVERABLE, **{c: R.FAILED_FLAG for c in temp_cols}},
        {"Image_Name": "MISSING_SCAN_RH", **{c: R.FAILED_FLAG for c in temp_cols}},
    ])
    summary_csv = tmp_path / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    monkeypatch.setattr(R, "SUMMARY_CSV", str(summary_csv))
    monkeypatch.setattr(R, "RETRY_CSV", str(tmp_path / "retry.csv"))
    monkeypatch.setattr(R, "MERGED_CSV", str(tmp_path / "merged.csv"))
    monkeypatch.setattr(sys, "argv", ["retry_failed_marker.py"])

    R.main()

    retry = pd.read_csv(tmp_path / "retry.csv").set_index("Image_Name")
    assert retry.loc[KNOWN_RECOVERABLE, "Retry_Status"] == "RECOVERED"
    assert retry.loc["MISSING_SCAN_RH", "Retry_Status"] == "FILE NOT FOUND"

    merged = pd.read_csv(tmp_path / "merged.csv").set_index("Image_Name")
    assert list(merged["Detection_Method"]) == ["MEDIAPIPE_DEFAULT", "MARKER_ANCHOR_RETRY", "FAILED"]
    recovered_temps = merged.loc[KNOWN_RECOVERABLE, temp_cols].astype(float)
    assert recovered_temps.notna().all()
    assert recovered_temps.between(15, 40).all()  # plausible skin temperatures
