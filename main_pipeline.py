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
    "D3_Distal":       [11, 12],  # DIP to Fingertip (includes nail region)
    
    # Digit 5 (Pinky Finger)
    "D5_Proximal":     [17, 18],  # MCP to PIP joint
    "D5_Intermediate": [18, 19],  # PIP to DIP joint
    "D5_Distal":       [19, 20]   # DIP to Fingertip (includes nail region)
}

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

def get_segment_avg_temp(raw_celsius_matrix: np.ndarray, landmarks, landmark_indices, opt_shape, padding=6):
    opt_h, opt_w = opt_shape
    therm_h, therm_w = raw_celsius_matrix.shape
    scale_x, scale_y = therm_w / opt_w, therm_h / opt_h

    # Extract pixel locations for the segment joints
    xs = [landmarks[idx].x * opt_w for idx in landmark_indices]
    ys = [landmarks[idx].y * opt_h for idx in landmark_indices]

    # Calculate padded bounding box around the segment
    ox1, ox2 = max(0, min(xs) - padding), min(opt_w, max(xs) + padding)
    oy1, oy2 = max(0, min(ys) - padding), min(opt_h, max(ys) + padding)

    # Convert optical coordinates to thermal matrix coordinates
    tx1, tx2 = int(ox1 * scale_x), int(ox2 * scale_x)
    ty1, ty2 = int(oy1 * scale_y), int(oy2 * scale_y)

    # Slice raw thermal matrix
    segment_matrix = raw_celsius_matrix[ty1:ty2, tx1:tx2]

    if segment_matrix.size == 0:
        return np.nan

    return float(np.mean(segment_matrix))

def process_scan(file_path: str):
    image_name = Path(file_path).stem
    
    try:
        thermogram = flyr.unpack(file_path)
        raw_celsius = thermogram.celsius
        
        if hasattr(thermogram, "optical") and thermogram.optical is not None:
            optical_image = thermogram.optical
        else:
            print(f"[{image_name}] No optical stream found. Skipping...")
            return
            
    except Exception as e:
        print(f"[{image_name}] Could not unpack file ({e}). Skipping...")
        return

    # Convert to RGB uint8 for MediaPipe Image format
    if len(optical_image.shape) == 3 and optical_image.shape[2] == 3:
        optical_rgb = cv2.cvtColor(optical_image, cv2.COLOR_BGR2RGB) if optical_image.dtype != np.uint8 else optical_image
    else:
        optical_rgb = optical_image

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=optical_rgb)
    detection_result = landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        print(f"[{image_name}] No hand detected by AI.")
        return

    hand_landmarks = detection_result.hand_landmarks[0]
    opt_shape = optical_rgb.shape[:2]

    summary_row: dict[str, str | float | None] = {"Image_Name": image_name}

    for segment_name, landmark_indices in PHALANX_LANDMARKS.items():
        avg_temp = get_segment_avg_temp(
            raw_celsius, hand_landmarks, landmark_indices, opt_shape
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
                    successful_detections += 1

    print(f"Total Scans Found:       {total_images_found}")
    print(f"Successful Detections:   {successful_detections}")
    print(f"Failed / Skipped Scans:  {total_images_found - successful_detections}")

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
        print(f"Saved detailed summary to: {output_csv}\n")

if __name__ == "__main__":
    main()
