import os
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import flyr
import mediapipe as mp

MODEL_PATH = "hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Missing '{MODEL_PATH}'!"
    )

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

PHALANX_LANDMARKS = {
    # Digit 3 (Middle Finger)
    "D3_Proximal":     [9, 10],   # MCP to PIP joint
    "D3_Intermediate": [10, 11],  # PIP to DIP joint
    "D3_Distal":       [11, 12],  # DIP to Fingertip
    
    # Digit 5 (Pinky Finger)
    "D5_Proximal":     [17, 18],  # MCP to PIP joint
    "D5_Intermediate": [18, 19],  # PIP to DIP joint
    "D5_Distal":       [19, 20]   # DIP to Fingertip
}

# Sampling band width as a fraction of the estimated finger width. Finger width
# is estimated from the index-to-pinky knuckle distance (landmarks 5 -> 17),
# which spans roughly 3 finger widths. Keeping the band narrower than the finger
# stops background/neighbouring-finger pixels from bleeding into the average.
PALM_LANDMARKS = (5, 17)
FINGER_WIDTHS_ACROSS_PALM = 3.0
BAND_WIDTH_FRACTION = 0.35
MIN_BAND_WIDTH_PX = 6  # floor for side-on hands, where knuckles overlap

THERMAL_INPUT_DIR = "raw_thermal_scans"
OUTPUT_DIR = "digit_phalanx_results"

def extract_zip_files():
    zip_files = ["LHand(1).zip", "RHand(1).zip"]
    for zip_name in zip_files:
        if os.path.exists(zip_name):
            print(f"Extracting {zip_name} into '{THERMAL_INPUT_DIR}'...")
            with zipfile.ZipFile(zip_name, 'r') as zip_ref:
                zip_ref.extractall(THERMAL_INPUT_DIR)
            print(f"Finished extracting {zip_name}.")

def unrotate_normalized(u, v, rot_angle):
    """Map normalized (0-1) coords in a rotated image back to the original orientation."""
    if rot_angle == cv2.ROTATE_90_CLOCKWISE:
        return v, 1 - u
    if rot_angle == cv2.ROTATE_180:
        return 1 - u, 1 - v
    if rot_angle == cv2.ROTATE_90_COUNTERCLOCKWISE:
        return 1 - v, u
    return u, v


def optical_to_thermal(u, v, opt_w, opt_h, therm_w, pip):
    """
    Map normalized optical coords to thermal pixel coords using the FLIR
    picture-in-picture metadata. The thermal and optical cameras have different
    fields of view and a parallax offset (varies per image with focus distance),
    so this mirrors the placement math in flyr's picture_in_picture_pil.
    """
    crop_x1, crop_y1, crop_x2, crop_y2 = pip.crop_box
    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

    # Size and top-left position of the thermal footprint inside the optical image
    scale = opt_w / therm_w / pip.real_to_ir
    dst_w, dst_h = round(crop_w * scale), round(crop_h * scale)
    origin_x = opt_w // 2 - dst_w // 2 + pip.offset_x
    origin_y = opt_h // 2 - dst_h // 2 + pip.offset_y

    tx = crop_x1 + (u * opt_w - origin_x) * crop_w / dst_w
    ty = crop_y1 + (v * opt_h - origin_y) * crop_h / dst_h
    return tx, ty


def estimate_band_width(palm_p1, palm_p2):
    """Band width in thermal pixels, scaled to the apparent size of the hand."""
    palm_width = np.hypot(palm_p2[0] - palm_p1[0], palm_p2[1] - palm_p1[1])
    finger_width = palm_width / FINGER_WIDTHS_ACROSS_PALM
    return max(MIN_BAND_WIDTH_PX, BAND_WIDTH_FRACTION * finger_width)


def segment_mask(shape, p1, p2, band_width):
    """Rectangular mask of the given width between two joints (no rounded end caps)."""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    direction = p2 - p1
    length = np.linalg.norm(direction)

    mask = np.zeros(shape, dtype=np.uint8)
    if length == 0:
        return mask

    normal = np.array([-direction[1], direction[0]]) / length * (band_width / 2)
    corners = np.array([p1 + normal, p2 + normal, p2 - normal, p1 - normal])
    cv2.fillConvexPoly(mask, np.round(corners).astype(np.int32), 255)
    return mask


def get_segment_avg_temp(raw_celsius_matrix: np.ndarray, p1, p2, band_width):
    therm_h, therm_w = raw_celsius_matrix.shape

    # Reject segments whose joints fall outside the thermal field of view
    for x, y in (p1, p2):
        if not (0 <= x < therm_w and 0 <= y < therm_h):
            return np.nan

    # Rectangular band following the finger angle between the two joints
    mask = segment_mask((therm_h, therm_w), p1, p2, band_width)

    # Extract thermal pixel values ONLY under the mask line
    segment_pixels = raw_celsius_matrix[mask == 255]

    if segment_pixels.size == 0:
        return np.nan

    return float(np.mean(segment_pixels))


def process_scan(file_path: str):
    image_name = Path(file_path).stem
    
    try:
        thermogram = flyr.unpack(file_path)
        raw_celsius = thermogram.celsius

        if hasattr(thermogram, "optical") and thermogram.optical is not None:
            optical_image = thermogram.optical
        else:
            print(f"[{image_name}] No optical stream found. Skipping...")
            return None

        pip = thermogram.pip_info
        if pip is None:
            print(f"[{image_name}] No optical/thermal alignment metadata found. Skipping...")
            return None
            
    except Exception as e:
        print(f"[{image_name}] Could not unpack file ({e}). Skipping...")
        return None

    # Exact image formatting from baseline script
    if len(optical_image.shape) == 3 and optical_image.shape[2] == 3:
        optical_rgb = cv2.cvtColor(optical_image, cv2.COLOR_BGR2RGB) if optical_image.dtype != np.uint8 else optical_image
    else:
        optical_rgb = optical_image

    # Rotation loop (0°, 90
    rotations = [
        None,                             # 0° (Original orientation)
        cv2.ROTATE_90_CLOCKWISE,          # 90°
        cv2.ROTATE_180,                   # 180°
        cv2.ROTATE_90_COUNTERCLOCKWISE    # 270°
    ]

    detection_result = None
    detected_rotation = None

    # Only the optical image is rotated for detection; landmarks are mapped back
    # to the original orientation before projecting onto the thermal grid.
    for rot_angle in rotations:
        current_opt = cv2.rotate(optical_rgb, rot_angle) if rot_angle is not None else optical_rgb

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_opt)
        res = landmarker.detect(mp_image)

        if res.hand_landmarks:
            detection_result = res
            detected_rotation = rot_angle
            break #Hand found!

    # FALLBACK LOGGING FOR FAILED DETECTIONS
    if not detection_result or not detection_result.hand_landmarks:
        print(f"[{image_name}] No hand detected by AI (tried all 4 rotations).")
        failed_row: dict[str, str] = {"Image_Name": image_name}
        for segment_name in PHALANX_LANDMARKS.keys():
            failed_row[f"{segment_name}_Avg_C"] = "NO HAND DETECTED - REVIEW IMAGE"
        return failed_row

    hand_landmarks = detection_result.hand_landmarks[0]
    summary_row: dict[str, str | float | None] = {"Image_Name": image_name}

    opt_h, opt_w = optical_rgb.shape[:2]
    therm_w = raw_celsius.shape[1]

    def landmark_to_thermal(idx):
        u, v = unrotate_normalized(hand_landmarks[idx].x, hand_landmarks[idx].y, detected_rotation)
        return optical_to_thermal(u, v, opt_w, opt_h, therm_w, pip)

    band_width = estimate_band_width(*(landmark_to_thermal(i) for i in PALM_LANDMARKS))

    for segment_name, (idx1, idx2) in PHALANX_LANDMARKS.items():
        avg_temp = get_segment_avg_temp(
            raw_celsius, landmark_to_thermal(idx1), landmark_to_thermal(idx2), band_width
        )
        summary_row[f"{segment_name}_Avg_C"] = round(avg_temp, 2) if not np.isnan(avg_temp) else None

    print(f"[{image_name}] Processed all phalanx segments successfully.")
    return summary_row


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(THERMAL_INPUT_DIR, exist_ok=True)
    
    # Extract archive files
    extract_zip_files()

    all_results = []
    total_images_found = 0
    successful_detections = 0

    # Process scans 
    for root, _, files in os.walk(THERMAL_INPUT_DIR):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".seq", ".tiff", ".flir")):
                total_images_found += 1
                file_path = os.path.join(root, file)
                
                res = process_scan(file_path)
                if res:
                    all_results.append(res)
                    # Count as success only if hand was detected
                    if res.get("D3_Proximal_Avg_C") != "NO HAND DETECTED - REVIEW IMAGE":
                        successful_detections += 1

    print(f"\nTotal Scans Found:       {total_images_found}")
    print(f"Successful Detections:   {successful_detections}")
    print(f"Failed / Flagged Scans:  {total_images_found - successful_detections}")

    if total_images_found > 0:
        success_rate = (successful_detections / total_images_found) * 100
        print(f"Detection Success Rate:  {success_rate:.2f}%")
    else:
        print("Detection Success Rate:  0.00% (No valid image files found)")

    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        output_csv = os.path.join(OUTPUT_DIR, "phalanx_temperatures_summary.csv")
        df.to_csv(output_csv, index=False)
        print(f"\nSaved detailed summary to: {output_csv}\n")

if __name__ == "__main__":
    main()