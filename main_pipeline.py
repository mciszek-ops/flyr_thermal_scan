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


def get_segment_avg_temp(raw_celsius_matrix: np.ndarray, landmarks, landmark_indices, opt_shape, line_thickness=16):


    therm_h, therm_w = raw_celsius_matrix.shape


    # Map normalized MediaPipe coordinates (0.0 to 1.0) directly to thermal array grid
    idx1, idx2 = landmark_indices
    x1, y1 = int(landmarks[idx1].x * therm_w), int(landmarks[idx1].y * therm_h)
    x2, y2 = int(landmarks[idx2].x * therm_w), int(landmarks[idx2].y * therm_h)


    # Create a blank mask matching thermal array shape
    mask = np.zeros((therm_h, therm_w), dtype=np.uint8)


    # Draw a thick line following the finger angle between the two joints
    cv2.line(mask, (x1, y1), (x2, y2), color=255, thickness=line_thickness)


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
           
    except Exception as e:
        print(f"[{image_name}] Could not unpack file ({e}). Skipping...")
        return None


    # Convert to RGB uint8 for MediaPipe Image format
    if len(optical_image.shape) == 3 and optical_image.shape[2] == 3:
        optical_rgb = cv2.cvtColor(optical_image, cv2.COLOR_BGR2RGB) if optical_image.dtype != np.uint8 else optical_image
    else:
        optical_rgb = optical_image


    # ==========================================================================
    # ROTATION FALLBACK LOOP (0°, 90°, 180°, 270°)
    # ==========================================================================
    rotations = [
        None,                             # 0° (Original)
        cv2.ROTATE_90_CLOCKWISE,          # 90°
        cv2.ROTATE_180,                   # 180°
        cv2.ROTATE_90_COUNTERCLOCKWISE    # 270°
    ]


    detection_result = None
    final_optical = optical_rgb
    final_thermal = raw_celsius


    for rot_angle in rotations:
        # Rotate both matrices together so spatial alignment is preserved
        current_opt = cv2.rotate(optical_rgb, rot_angle) if rot_angle is not None else optical_rgb
        current_therm = cv2.rotate(raw_celsius, rot_angle) if rot_angle is not None else raw_celsius


        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=current_opt)
        res = landmarker.detect(mp_image)


        if res.hand_landmarks:
            detection_result = res
            final_optical = current_opt
            final_thermal = current_therm
            break  # Stop trying angles once hand is detected


    if not detection_result or not detection_result.hand_landmarks:
        print(f"[{image_name}] No hand detected by AI (tried all 4 rotations).")
        return None


    # Extract detected landmarks and rotated shapes
    hand_landmarks = detection_result.hand_landmarks[0]
    opt_shape = final_optical.shape[:2]


    summary_row: dict[str, str | float | None] = {"Image_Name": image_name}


    for segment_name, landmark_indices in PHALANX_LANDMARKS.items():
        avg_temp = get_segment_avg_temp(
            final_thermal, hand_landmarks, landmark_indices, opt_shape, line_thickness=16
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
