# drift_detection_system
🔍 Overview

This project implements a complete Data & Concept Drift Detection Pipeline designed for real-world machine learning systems where data changes over time can silently degrade model performance.

The system monitors streaming or batch data, identifies different types of drift (linear, non-linear, and statistical anomalies), logs drift events into a SQL database, triggers alerts, and supports proactive retraining.

This project is architected using:

PCA (Principal Component Analysis) → detects linear drift

LSTM Autoencoder → detects non-linear drift

Statistical Tests → detects distribution drift

SQL Logging Pipeline

Visualizer for Drift Reports

It’s built to demonstrate strong ML engineering skills and real-world model maintenance principles.

🚀 Key Features
1. Multi-Model Drift Detection

The system uses three independent techniques to detect drift:

🔹 PCA Reconstruction Error

Good for linear pattern changes

Detects shifts in directional variance

Fast and efficient

🔹 LSTM Autoencoder Reconstruction Loss

Captures non-linear, temporal patterns

Best for time-series or sequential drift

Resistant to noise and sudden spikes

🔹 Statistical Drift Tests

Kolmogorov–Smirnov (KS Test)

Population Stability Index (PSI)
Used to detect distribution-level changes between reference and new data.

2. Automated Event Logging

Drift events are saved with:

Timestamp

Model used

Reconstruction error / KS score / PSI score

Drift severity

Drift type (linear, nonlinear, statistical)

Database backend:

Primary: MySQL

Fallback: SQLite (auto-switch)

3. Monitoring Pipeline

Included:

Windowing-based time series processor

Normalization using StandardScaler

Real-time drift percentage calculator

Threshold-based alerting

4. Visual Drift Report Generator

Visual outputs include:

Reconstruction error plots

KS distribution graphs

PSI trend charts

🏗️ Project Structure
drift_system/
│
├── drift/
│   ├── drift_detector.py       # Core drift detection engine
│   └── windowing.py            # Sliding window generator
│
├── models/
│   ├── pca_model.pkl           # Trained PCA model
│   ├── lstm_model.h5           # Trained LSTM Autoencoder
│   └── scalers.pkl             # StandardScaler
│
├── stats/
│   └── statistical_tests.py    # KS test, PSI test implementations
│
├── logger/
│   ├── mysql_logger.py         # MySQL + SQLite hybrid logger
│   └── logger.py               # Logging wrapper
│
├── visualizer/
│   └── visualizer.py           # Drift visualization utilities
│
├── data/
│   └── simulated_time_series.csv
│
└── demo_run.py                 # Full end-to-end demo script

⚙️ How It Works — Architecture
Step 1 — Preprocessing

Normalization

Sliding window creation

Temporal batch segmentation

Step 2 — Drift Detection

Each incoming batch is compared against reference data:

🔸 PCA

Reconstructs data → calculates error → compares with threshold.

🔸 LSTM Autoencoder

Learns normal patterns → error spikes = drift.

🔸 Statistical Tests

KS Test → compares CDFs

PSI → compares bin distributions

Step 3 — Drift Flagging

If error/stat score > threshold:

Drift event created

Drift severity computed

Logged in SQL database

Step 4 — Visualization

Generate charts for:

RE distribution

Drift timestamps

Statistical scores

🛠️ Installation
pip install -r requirements.txt


Ensure MySQL service is running if using MySQL logger.

▶️ Running the System
python demo_run.py


This:

Loads models

Simulates time series data

Injects artificial drift

Runs PCA + LSTM + statistical tests

Logs events

Generates visual drift reports

📚 Why Multi-Model Drift Detection?

Different drifts require different detectors:

Drift Type	Example	Best Detector
Linear Drift	Gradually increasing values	PCA
Non-Linear Drift	Sudden pattern breaks	LSTM Autoencoder
Distribution Drift	Value ranges change	KS Test / PSI

Using multiple detectors ensures higher reliability, especially for real-world noisy data.

📈 Real-World Use Cases

Fraud detection systems

Banking KYC monitoring

Anomaly detection in IoT sensors

Retail demand forecasting

Healthcare patient monitoring

Any ML model deployed in production

🧠 Math Concepts Used
PCA

Covariance matrix

Eigenvalues/eigenvectors

Explained variance

Reconstruction error

LSTM

Cell state & hidden state

Forget, input, output gates

Vanishing gradient prevention

Sequence autoencoding

Statistical Tests

KS statistic & CDF

PSI formula

Joint distribution comparison

Interview-ready explanation included.

📝 Future Enhancements

Online learning support

Auto retraining scheduler

Drift root cause analysis (RCA)

Grafana or Streamlit dashboards

💡 Author

Cyril — Data Scientist & ML Engineer
Aiming FAANG/MAANG-ready ML Ops and ML Engineering mastery.
