
# Wi-Fi Fingerprinting and Barometric Pressure Fusion for Floor Identification

## Overview

This project explores indoor floor identification and localization using Wi-Fi fingerprinting and smartphone barometric pressure measurements.

Since GPS signals are often unreliable indoors, the system uses Wi-Fi signal fingerprints to estimate horizontal position (X, Y) and floor level. Barometric pressure is also investigated as a source of vertical information (Z).

The project evaluates Wi-Fi-based methods, pressure-based floor estimation, and sensor fusion strategies.

## Dataset

The experiments use the **IPIN 2025 Competition Track 3 dataset**, collected in the Paatalo building in Finland.

The dataset contains measurements from floors 2–6 and includes:

- Wi-Fi access point identifiers (BSSID)
- Wi-Fi signal strength (RSSI)
- Frequency and timestamps
- Barometric pressure measurements
- Ground-truth coordinates and floor labels
- Trial and device identifiers

After removing a trial with corrupted pressure measurements, the main experiment uses 760 samples from 17 trials.

**Dataset availability:** The original dataset is not included in this repository. Obtain it from the official IPIN 2025 dataset source, subject to its access and usage conditions.

## Methods

### 1. Wi-Fi Fingerprinting

Wi-Fi fingerprints are compared with reference fingerprints collected at known locations.

Two search approaches are evaluated:

- **k-Nearest Neighbours (kNN):** Compares a query fingerprint against the training fingerprints and uses the five nearest neighbours for estimation.
- **STH-style search:** Uses K-means clustering and BallTree-based search to reduce the number of candidate fingerprints before applying nearest-neighbour estimation.

### 2. Barometric Pressure Estimation

Pressure measurements are analysed to investigate their relationship with floor changes.

The experiments consider pressure smoothing, device-dependent calibration differences, starting-floor anchors, and pressure-based floor estimation.

### 3. Sensor Fusion

Two fusion approaches are evaluated:

- **Naive Bayesian Fusion:** Combines Wi-Fi and pressure-based floor probabilities.
- **Gated Fusion:** Uses pressure information when the absolute pressure change reaches the selected threshold; otherwise, it retains the Wi-Fi estimate.

## Experimental Setup

- Building: Paatalo, Finland
- Dataset: IPIN 2025 Competition Track 3
- Floors: 2–6
- Devices: SA52 and SG24
- Main dataset: 760 samples across 17 trials
- Test trials: 21, 25, 34, and 44, and 54
- kNN neighbours: 5
- Gated fusion pressure threshold: 0.15 hPa

## Results

### Main Experiment: Floors 2–6

| Method | Floor Accuracy | Macro F1 | XY RMSE |
|---|---:|---:|---:|
| Wi-Fi kNN | 69.0% | 0.732 | 10.46 m |
| STH-style search | 69.9% | 0.740 | 11.57 m |
| Pressure-only (Wi-Fi anchor) | 25.0% | 0.207 | — |
| Pressure-only (ground-truth anchor) | 40.3% | 0.280 | — |
| Naive Bayesian Fusion | 64.3% | 0.649 | — |
| Gated Fusion | 67.6% | 0.682 | — |

### Key Findings

- STH-style search slightly improves floor accuracy and Macro F1 compared with the kNN baseline.
- STH-style search reduces candidate comparisons from 544 to 5 on average, but does not achieve lower measured execution time in this dataset.
- Pressure-based estimation is affected by device calibration differences and atmospheric variation.
- Gated fusion performs better than naive Bayesian fusion.
- Neither fusion method outperforms Wi-Fi-only kNN in the main experiment.

## Technologies Used

- Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Jupyter Notebook

## Repository Contents

- `anchor.ipynb` — Main experimental notebook containing the implementation and analysis.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.

## How to Run

1. Clone or download this repository.
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download and extract the IPIN 2025 Competition Track 3 dataset, following its usage conditions.
4. Update the dataset folder path in `anchor.ipynb` to point to your extracted dataset.
5. Open the notebook using Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `anchor.ipynb` and execute the cells in order.

## Limitations

- Experiments are conducted using data from a single building.
- The number of trials and devices is limited.
- Barometric pressure measurements are sensitive to calibration differences and atmospheric drift.
- Pressure-based estimation depends on the accuracy of the starting-floor anchor.
- The tested fusion approaches do not outperform the Wi-Fi-only kNN baseline.

## Acknowledgements

The experiments use the IPIN 2025 Competition Track 3 dataset.
