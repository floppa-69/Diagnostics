
# Engine Anomaly Detector

A compact repository containing a TensorFlow Lite model and helper scripts for detecting engine anomalies from vehicle sensor data. This README focuses on quick integration (including Android), usage, and notes about preprocessing and model behavior.

## Quick start

Prerequisites:
- Python 3.8+ with packages: numpy, pandas, tensorflow, matplotlib, seaborn (used by `test.py`).

Run the detection demo (interactive):

```bash
python3 test.py
```

When prompted, enter the path to your CSV file (or press Enter to use the default filename shown by the script). Required files in the working directory:
- `engine_anomaly_detector.tflite` (TFLite model)
- `scaler.pkl` (scikit-learn scaler used for normalization)
- `model_metadata.pkl` (feature order, sequence length, class names)

The script will run inference, show plots, and write `detection_report.txt` and `analysis_results.png`.

## Files in this repo

- `engine_anomaly_detector.tflite` — The TensorFlow Lite model for inference.
- `scaler.pkl` — Scaler used to normalize features (required at runtime).
- `model_metadata.pkl` — Metadata: `feature_columns`, `sequence_length`, `class_names`.
- `test.py` — Demo/analysis script: loads model & scaler, runs inference on CSVs, visualizes results, and writes a report.
- `train.py` — Training/experimentation scripts (if you want to retrain or improve the model).
- `detection_report.txt` — Example generated report (output of `test.py`).

## Model specification

- Input shape: [1, 10, 14] — batch size 1, sequence length 10, 14 features per timestep.
- Data type: float32
- Output: array-like shape [1, 5] — probability per class. Choose argmax for predicted class.

Classes (index -> label):

0 — Normal
1 — Weak Injectors
2 — Fuel Leak
3 — Oil Leak
4 — Vacuum Leak

## Feature order (required)

The model expects features in the exact order defined in `model_metadata.pkl` (the example training order included in this repo). When you preprocess CSVs use the `feature_columns` list from the metadata to ensure column order.

Typical features include (example names):
- average_speed, barometric_pressure, calculated_boost, engine_load, distance_travelled,
  engine_coolant_temp, engine_rpm, engine_rpm_x1000, intake_manifold_pressure,
  long_term_fuel_trim_bank1, obd_voltage, short_term_fuel_trim_bank1, vehicle_acceleration, vehicle_speed

Always rely on `model_metadata.pkl` to programmatically obtain the exact column names.

## Preprocessing (critical)

The model was trained on standardized input. You must apply the same scaling at inference time.

Example (Python, using the provided `scaler.pkl`):

```python
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# X is a (N, 14) raw-feature array extracted with model_metadata['feature_columns']
X_scaled = scaler.transform(X)
```

For mobile/Android, extract the scaler statistics (mean & scale) and apply:

normalized = (raw - mean) / scale

Do not pass raw sensor values into the model — predictions will be invalid.

## Android integration tips

- Place `engine_anomaly_detector.tflite` into your app's `assets/` folder.
- Ensure you prepare an input buffer shaped [1, 10, 14] (float32). Fill it with recent 10 timesteps in exact feature order.
- Use the TFLite Interpreter API, call `interpreter.allocateTensors()`, set input tensor, `invoke()`, and read the output tensor.
- Convert the model output to a class by selecting the index with the highest probability.

Small pseudo-steps:
1. Maintain a FIFO buffer of the last 10 feature rows.
2. Apply the same normalization per-feature.
3. Build a float32 input array: shape [1, 10, 14].
4. Run inference and interpret results.

## Running locally (examples)

Run interactive analysis (default behavior of `test.py`):

```bash
python3 test.py
# then enter the CSV path when prompted (or press Enter for the default)
```

If you prefer non-interactive usage, open `test.py` and adapt the `csv_path` variable in `main()` or modify the script to accept a CLI argument.

## Recommendations & app logic

- Aggregate predictions over a window (e.g., 60s) to compute an anomaly rate.
- Suggested thresholds (used in `test.py`):
  - Anomaly rate < 5% — Healthy
  - 5% ≤ Anomaly rate < 20% — Warning
  - Anomaly rate ≥ 20% — Critical

## Contact & license

This repository is provided as-is for research/demo purposes. Check project root for license information (or add a LICENSE file if you redistribute).

## Notes

- Keep `scaler.pkl` and `model_metadata.pkl` together with the TFLite model for reliable inference.
- If you retrain the model, export and update `scaler.pkl` and `model_metadata.pkl` accordingly.
