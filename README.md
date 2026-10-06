# flyr_thermal_scan
Batch-processing Python script to detect hand landmarks across a 3,000+ thermal images. 

# FLIR Thermal Phalanx Extraction Pipeline

A production-ready Python pipeline for extracting mean temperatures (°C) from specific finger phalanx segments (Digits 3 & 5) using FLIR radiometric images.

## Features
- Radiometric unpacking via `flyr`
- Landmark detection with `MediaPipe Tasks Vision API`
- Oriented line-masking (capsule mask) for diagonal finger sampling
- 4-angle rotation fallback loop (0°, 90°, 180°, 270°) for rotated scans

## Requirements
- Python 3.10+
- `hand_landmarker.task` model file placed in the root directory

## Retrying failed scans with the marker anchor
`retry_failed_marker.py` re-processes scans flagged `NO HAND DETECTED` in the main summary. It finds the circular marker sticker on the back of the hand (in the thermal image), re-runs MediaPipe on marker-centred crops at 30° rotation steps, and only accepts hands whose palm contains the marker, whose size matches the marker, and whose finger order matches the LH/RH filename.

```
python main_pipeline.py            # produces phalanx_temperatures_summary.csv
python retry_failed_marker.py      # add --debug to save overlay images for QA
```
Outputs `marker_retry_results.csv` and `phalanx_temperatures_summary_with_retry.csv` (with a `Detection_Method` column) in `digit_phalanx_results/`.
