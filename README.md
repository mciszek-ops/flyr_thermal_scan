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
