"""
Retry scans that failed hand detection in main_pipeline.py, using the circular
marker sticker on the back of the hand as an anchor.

MediaPipe's palm detector anchors on the palm/wrist and only sees the image at
the 4 right-angle rotations the main pipeline tries, so it misses hands that are
tilted at odd angles, small in frame, or partly cropped. Every scan has a
circular sticker on the tape across the back of the hand, which shows up in the
thermal image as a crisp disc with uniform tape around it. This script:

  1. Finds the marker in the thermal image (Hough circles scored by how evenly
     the disc contrasts with the tape ring around it), and checks that the same
     spot is white in the optical image.
  2. Re-runs MediaPipe on crops centred on the marker, sized from the marker's
     radius, at 30-degree rotation steps, with lower confidence thresholds.
  3. Accepts a detection only if the marker lies on the detected back of the
     hand and the hand is a plausible size relative to the marker. This is what
     makes it safe to lower MediaPipe's confidence thresholds.
  4. Samples phalanx temperatures exactly as main_pipeline.py does.

Results are written to a separate CSV, plus a merged copy of the main summary
with recovered rows filled in. The original summary is never modified.

Usage:
    python retry_failed_marker.py            # retry every failed scan
    python retry_failed_marker.py --debug    # also save overlay images for QA
    python retry_failed_marker.py --limit 20 # quick test on the first 20 failures
"""
import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import flyr
import mediapipe as mp

from main_pipeline import (
    MODEL_PATH,
    OUTPUT_DIR,
    PALM_LANDMARKS,
    PHALANX_LANDMARKS,
    THERMAL_INPUT_DIR,
    estimate_band_width,
    get_segment_avg_temp,
    optical_to_thermal,
)

FAILED_FLAG = "NO HAND DETECTED - REVIEW IMAGE"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "phalanx_temperatures_summary.csv")
RETRY_CSV = os.path.join(OUTPUT_DIR, "marker_retry_results.csv")
MERGED_CSV = os.path.join(OUTPUT_DIR, "phalanx_temperatures_summary_with_retry.csv")
DEBUG_DIR = os.path.join(OUTPUT_DIR, "marker_retry_debug")

# Marker detection in the 320x240 thermal grid
MARKER_MIN_RADIUS_PX = 7
MARKER_MAX_RADIUS_PX = 24
# Candidates are scored min(thermal, 2) * whiteness^3 (calibrated on scans the
# default pipeline detected). The top few are tried in turn, since the hand
# validation step rejects a wrong marker anyway.
MARKER_MIN_SCORE = 0.25
MARKER_MAX_CANDIDATES = 3

# Crop half-size in multiples of the marker radius. The fingertips sit roughly
# 9-12 marker radii from the marker, so these crops frame the whole hand at a
# few zoom levels.
CROP_HALF_SIZES = (11.0, 14.0, 8.5)
CROP_ROTATIONS_DEG = range(0, 360, 30)
CROP_OUTPUT_PX = 512

# Relaxed thresholds: the marker-based validation below rejects false positives
RETRY_DETECTION_CONFIDENCE = 0.2
RETRY_PRESENCE_CONFIDENCE = 0.2

# Plausible knuckle width (landmarks 5 -> 17) in marker radii
KNUCKLE_WIDTH_RANGE = (3.0, 10.0)
# The marker may sit slightly outside the knuckle/wrist polygon (it is on the
# back of the hand near the wrist), so the polygon is grown by this many radii
PALM_POLYGON_MARGIN_R = 1.5
PALM_POLYGON_LANDMARKS = [0, 1, 5, 9, 13, 17]

# Hands are photographed back-up, so the finger order fixes the sign of
# cross(finger direction, index->pinky knuckles) in image coordinates. On scans
# the default pipeline detected this was negative for 99% of left hands and
# positive for 91% of right hands. A wrong sign means MediaPipe has mirrored the
# finger order (e.g. labelled the index finger as the pinky).
EXPECTED_CHIRALITY = {"LH": -1, "RH": 1}

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_retry_landmarker():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=RETRY_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=RETRY_PRESENCE_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


def thermal_to_optical(tx, ty, opt_w, opt_h, therm_w, pip):
    """Inverse of main_pipeline.optical_to_thermal, returning optical pixel coords."""
    crop_x1, crop_y1, crop_x2, crop_y2 = pip.crop_box
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

    scale = opt_w / therm_w / pip.real_to_ir
    dst_w, dst_h = round(crop_w * scale), round(crop_h * scale)
    origin_x = opt_w // 2 - dst_w // 2 + pip.offset_x
    origin_y = opt_h // 2 - dst_h // 2 + pip.offset_y

    ox = origin_x + (tx - crop_x1) * dst_w / crop_w
    oy = origin_y + (ty - crop_y1) * dst_h / crop_h
    return ox, oy, dst_w / crop_w


def ring_samples(image, x, y, radius, n=48):
    """Sample image values on a circle; also return which samples were in bounds."""
    h, w = image.shape[:2]
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    px, py = x + radius * np.cos(ang), y + radius * np.sin(ang)
    inside = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    px = np.clip(np.round(px).astype(int), 0, w - 1)
    py = np.clip(np.round(py).astype(int), 0, h - 1)
    return image[py, px], inside


def score_marker_candidate(celsius, x, y, r):
    """
    A real marker is a uniform disc whose edge contrasts with the tape in the same
    direction all the way round. Clutter (mesh, folds, finger gaps) contrasts only
    on some sides or is not uniform inside.
    """
    disc = np.concatenate([ring_samples(celsius, x, y, r * f)[0] for f in (0.0, 0.3, 0.55)])
    inner, ok_in = ring_samples(celsius, x, y, r * 0.7)
    outer, ok_out = ring_samples(celsius, x, y, r * 1.3)
    ok = ok_in & ok_out
    if ok.mean() < 0.6:
        return 0.0

    diff = (outer - inner)[ok]
    median = np.median(diff)
    if abs(median) < 0.2:
        return 0.0

    consistency = np.mean((np.sign(diff) == np.sign(median)) & (np.abs(diff) > 0.4 * abs(median)))
    uniformity = abs(median) / (disc.std() + 0.1)
    return float(consistency ** 3 * min(uniformity, 4.0))


def optical_whiteness(optical_rgb, x, y, r):
    """How sticker-like (bright, unsaturated) a disc of the optical image is, 0-1."""
    h, w = optical_rgb.shape[:2]
    if not (0 <= x < w and 0 <= y < h):
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (int(round(x)), int(round(y))), max(2, int(r * 0.6)), 255, -1)
    hsv = cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[..., 1][mask > 0].mean() / 255
    val = hsv[..., 2][mask > 0].mean() / 255
    img_val = np.percentile(hsv[..., 2], 90) / 255
    return float(np.clip(1 - sat * 2, 0, 1) * np.clip(val / max(img_val, 1e-3), 0, 1))


def find_marker_candidates(celsius, optical_rgb, pip):
    """
    Locate likely markers, best first. Each is a dict with thermal + optical
    position/radius and score. Empty if no convincing candidate was found.
    """
    lo, hi = np.percentile(celsius, 1), np.percentile(celsius, 99)
    norm = (np.clip((celsius - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    norm = cv2.GaussianBlur(norm, (3, 3), 0)
    circles = cv2.HoughCircles(
        norm, cv2.HOUGH_GRADIENT, dp=1, minDist=6, param1=40, param2=10,
        minRadius=MARKER_MIN_RADIUS_PX, maxRadius=MARKER_MAX_RADIUS_PX,
    )
    if circles is None:
        return []

    smooth = cv2.GaussianBlur(celsius.astype(np.float32), (3, 3), 0)
    opt_h, opt_w = optical_rgb.shape[:2]
    therm_w = celsius.shape[1]

    candidates = []
    for tx, ty, tr in circles[0]:
        thermal_score = score_marker_candidate(smooth, tx, ty, tr)
        if thermal_score <= 0:
            continue
        ox, oy, px_scale = thermal_to_optical(tx, ty, opt_w, opt_h, therm_w, pip)
        orad = tr * px_scale
        white = optical_whiteness(optical_rgb, ox, oy, orad)
        score = min(thermal_score, 2.0) * white ** 3
        if score >= MARKER_MIN_SCORE:
            candidates.append({
                "thermal_xy": (float(tx), float(ty)), "thermal_r": float(tr),
                "optical_xy": (float(ox), float(oy)), "optical_r": float(orad),
                "thermal_score": thermal_score, "whiteness": white, "score": score,
            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:MARKER_MAX_CANDIDATES]


def marker_crop_transform(center, half_size, angle_deg):
    """Affine map from optical pixels to a square crop centred on the marker."""
    scale = CROP_OUTPUT_PX / (2 * half_size)
    m = cv2.getRotationMatrix2D(center, angle_deg, scale)
    m[0, 2] += CROP_OUTPUT_PX / 2 - center[0]
    m[1, 2] += CROP_OUTPUT_PX / 2 - center[1]
    return m


def hand_chirality(points_px):
    """+1 or -1 depending on which side of the finger axis the pinky lies."""
    axis = points_px[[8, 12, 16, 20]].mean(axis=0) - points_px[[5, 9, 13, 17]].mean(axis=0)
    across = points_px[17] - points_px[5]
    return int(np.sign(axis[0] * across[1] - axis[1] * across[0]))


def validate_against_marker(points_px, marker, expected_chirality=None):
    """
    Check a hand (21 landmarks in optical pixels) is consistent with the marker:
    plausible size relative to the sticker, the sticker sits on the back of the
    hand, and the finger order matches the hand side from the filename.
    Returns a penalty (lower is better) or None if rejected.
    """
    if expected_chirality and hand_chirality(points_px) != expected_chirality:
        return None

    r = marker["optical_r"]
    mx, my = marker["optical_xy"]

    knuckle_width = np.linalg.norm(points_px[PALM_LANDMARKS[0]] - points_px[PALM_LANDMARKS[1]])
    knuckle_ratio = knuckle_width / r
    if not (KNUCKLE_WIDTH_RANGE[0] <= knuckle_ratio <= KNUCKLE_WIDTH_RANGE[1]):
        return None

    palm = points_px[PALM_POLYGON_LANDMARKS].astype(np.float32)
    hull = cv2.convexHull(palm)
    dist = cv2.pointPolygonTest(hull, (float(mx), float(my)), True)  # +inside, -outside
    if dist < -PALM_POLYGON_MARGIN_R * r:
        return None

    # Prefer detections whose palm centre is close to the marker
    palm_centre = palm.mean(axis=0)
    return float(np.linalg.norm(palm_centre - (mx, my)) / r)


def detect_hand_with_marker(landmarker, optical_rgb, marker, expected_chirality=None):
    """
    Search marker-centred crops for a hand that passes validation.
    Returns (landmarks in optical pixels, info dict) or (None, None).
    """
    center = marker["optical_xy"]
    candidates = []

    for half_mult in CROP_HALF_SIZES:
        half_size = half_mult * marker["optical_r"]
        for angle in CROP_ROTATIONS_DEG:
            m = marker_crop_transform(center, half_size, angle)
            crop = cv2.warpAffine(
                optical_rgb, m, (CROP_OUTPUT_PX, CROP_OUTPUT_PX),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
            )
            res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop)))

            inv = cv2.invertAffineTransform(m)
            for hand_idx, hand in enumerate(res.hand_landmarks):
                crop_px = np.array([[lm.x * CROP_OUTPUT_PX, lm.y * CROP_OUTPUT_PX] for lm in hand])
                opt_px = crop_px @ inv[:, :2].T + inv[:, 2]
                penalty = validate_against_marker(opt_px, marker, expected_chirality)
                if penalty is None:
                    continue
                confidence = res.handedness[hand_idx][0].score if res.handedness else 0.0
                candidates.append((penalty - confidence, opt_px, {
                    "crop_half_size_r": half_mult, "crop_rotation_deg": angle,
                    "mp_confidence": round(float(confidence), 3), "marker_offset_r": round(penalty, 2),
                }))

        # A validated hand at this zoom level is good enough; skip the rest
        if candidates:
            break

    if not candidates:
        return None, None
    _, points, info = min(candidates, key=lambda c: c[0])
    return points, info


def save_debug_overlay(path, optical_rgb, marker, points_px):
    img = cv2.cvtColor(optical_rgb, cv2.COLOR_RGB2BGR)
    if marker:
        cx, cy = (int(round(v)) for v in marker["optical_xy"])
        cv2.circle(img, (cx, cy), int(marker["optical_r"]), (0, 255, 0), 3)
    if points_px is not None:
        for idx1, idx2 in mp.solutions.hands.HAND_CONNECTIONS:
            p1, p2 = points_px[idx1].astype(int), points_px[idx2].astype(int)
            cv2.line(img, tuple(p1), tuple(p2), (255, 255, 255), 2)
        for (idx1, idx2) in PHALANX_LANDMARKS.values():
            p1, p2 = points_px[idx1].astype(int), points_px[idx2].astype(int)
            cv2.line(img, tuple(p1), tuple(p2), (0, 0, 255), 5)
    cv2.imwrite(path, img)


def retry_scan(landmarker, file_path, debug=False):
    image_name = Path(file_path).stem
    row = {"Image_Name": image_name}

    try:
        thermogram = flyr.unpack(file_path)
        raw_celsius = thermogram.celsius
        optical_rgb = thermogram.optical
        pip = thermogram.pip_info
    except Exception as e:
        print(f"[{image_name}] Could not unpack file ({e}).")
        row["Retry_Status"] = "UNPACK FAILED"
        return row

    if optical_rgb is None or pip is None:
        row["Retry_Status"] = "NO OPTICAL/ALIGNMENT DATA"
        return row

    markers = find_marker_candidates(raw_celsius, optical_rgb, pip)
    if not markers:
        print(f"[{image_name}] Marker not found.")
        row["Retry_Status"] = "MARKER NOT FOUND"
        if debug:
            save_debug_overlay(os.path.join(DEBUG_DIR, f"{image_name}_nomarker.jpg"), optical_rgb, None, None)
        return row

    # Try each marker candidate in turn; validation rejects a wrong one
    expected_chirality = EXPECTED_CHIRALITY.get(image_name.upper().rsplit("_", 1)[-1])
    for marker in markers:
        points_px, info = detect_hand_with_marker(landmarker, optical_rgb, marker, expected_chirality)
        if points_px is not None:
            break
    else:
        marker = markers[0]

    row.update({
        "Marker_Thermal_X": round(marker["thermal_xy"][0], 1),
        "Marker_Thermal_Y": round(marker["thermal_xy"][1], 1),
        "Marker_Thermal_R": round(marker["thermal_r"], 1),
        "Marker_Score": round(marker["score"], 2),
    })

    if points_px is None:
        print(f"[{image_name}] Marker found but no hand validated against it.")
        row["Retry_Status"] = "NO HAND NEAR MARKER"
        if debug:
            save_debug_overlay(os.path.join(DEBUG_DIR, f"{image_name}_nohand.jpg"), optical_rgb, marker, None)
        return row

    row.update(info)
    opt_h, opt_w = optical_rgb.shape[:2]
    therm_w = raw_celsius.shape[1]

    def landmark_to_thermal(idx):
        u, v = points_px[idx][0] / opt_w, points_px[idx][1] / opt_h
        return optical_to_thermal(u, v, opt_w, opt_h, therm_w, pip)

    band_width = estimate_band_width(*(landmark_to_thermal(i) for i in PALM_LANDMARKS))
    for segment_name, (idx1, idx2) in PHALANX_LANDMARKS.items():
        avg_temp = get_segment_avg_temp(
            raw_celsius, landmark_to_thermal(idx1), landmark_to_thermal(idx2), band_width
        )
        row[f"{segment_name}_Avg_C"] = round(avg_temp, 2) if not np.isnan(avg_temp) else None

    row["Retry_Status"] = "RECOVERED"
    print(f"[{image_name}] Recovered via marker anchor "
          f"(zoom {info['crop_half_size_r']}r, rot {info['crop_rotation_deg']} deg).")
    if debug:
        save_debug_overlay(os.path.join(DEBUG_DIR, f"{image_name}_recovered.jpg"), optical_rgb, marker, points_px)
    return row


def find_scan_paths():
    paths = {}
    for root, _, files in os.walk(THERMAL_INPUT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".seq", ".tiff", ".flir")):
                paths[Path(file).stem] = os.path.join(root, file)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--debug", action="store_true", help=f"save overlay images to {DEBUG_DIR}")
    parser.add_argument("--limit", type=int, help="only retry the first N failed scans")
    args = parser.parse_args()

    if not os.path.exists(SUMMARY_CSV):
        raise FileNotFoundError(f"'{SUMMARY_CSV}' not found. Run main_pipeline.py first.")
    if args.debug:
        os.makedirs(DEBUG_DIR, exist_ok=True)

    summary = pd.read_csv(SUMMARY_CSV)
    flag_col = f"{next(iter(PHALANX_LANDMARKS))}_Avg_C"
    failed_names = summary.loc[summary[flag_col].astype(str) == FAILED_FLAG, "Image_Name"].tolist()
    if args.limit:
        failed_names = failed_names[:args.limit]

    scan_paths = find_scan_paths()
    landmarker = create_retry_landmarker()

    print(f"Retrying {len(failed_names)} failed scans using the marker as anchor...\n")
    rows = []
    for name in failed_names:
        if name not in scan_paths:
            print(f"[{name}] Scan file not found in '{THERMAL_INPUT_DIR}'.")
            rows.append({"Image_Name": name, "Retry_Status": "FILE NOT FOUND"})
            continue
        rows.append(retry_scan(landmarker, scan_paths[name], debug=args.debug))

    retry_df = pd.DataFrame(rows)
    retry_df.to_csv(RETRY_CSV, index=False)

    status_counts = retry_df["Retry_Status"].value_counts()
    recovered = int(status_counts.get("RECOVERED", 0))
    print("\nMarker retry results:")
    for status, count in status_counts.items():
        print(f"  {status:<28}{count:>6}")
    if len(retry_df):
        print(f"Recovered {recovered} of {len(retry_df)} failed scans ({recovered / len(retry_df) * 100:.2f}%).")

    # Merged summary: fill recovered rows in, mark where each row's values came from
    merged = summary.copy()
    merged["Detection_Method"] = np.where(
        merged[flag_col].astype(str) == FAILED_FLAG, "FAILED", "MEDIAPIPE_DEFAULT"
    )
    recovered_df = retry_df[retry_df["Retry_Status"] == "RECOVERED"].set_index("Image_Name")
    temp_cols = [f"{segment}_Avg_C" for segment in PHALANX_LANDMARKS]
    merged = merged.set_index("Image_Name")
    merged[temp_cols] = merged[temp_cols].astype(object)
    merged.loc[recovered_df.index, temp_cols] = recovered_df[temp_cols].values
    merged.loc[recovered_df.index, "Detection_Method"] = "MARKER_ANCHOR_RETRY"
    merged.reset_index().to_csv(MERGED_CSV, index=False)

    total = len(merged)
    ok = int((merged["Detection_Method"] != "FAILED").sum())
    print(f"\nOverall detection rate with retry: {ok}/{total} ({ok / total * 100:.2f}%)")
    print(f"Saved retry details to: {RETRY_CSV}")
    print(f"Saved merged summary to: {MERGED_CSV}\n")


if __name__ == "__main__":
    main()
