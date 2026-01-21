
import pandas as pd
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from drift.windowing import sliding_window
from drift.drift_detector import DriftDetector
from logger.logger import log_event

df = pd.read_csv("data/simulated_time_series.csv")
values = df.values

windows = sliding_window(values)

windows_pca = windows.reshape(windows.shape[0], -1)
pca = PCA(n_components=1).fit(windows_pca)

lstm = Sequential([
    LSTM(10, return_sequences=True, input_shape=(windows.shape[1],1)),
    Dense(1)
])
lstm.compile(optimizer="adam", loss="mse")
lstm.fit(windows.reshape(-1, windows.shape[1],1), windows.reshape(-1, windows.shape[1],1), epochs=2)

detector = DriftDetector(pca, lstm)

pca_err = detector.detect_pca_drift(windows_pca)
lstm_err = detector.detect_lstm_drift(windows.reshape(-1, windows.shape[1],1))

event = {
    "pca_error": pca_err,
    "lstm_error": lstm_err
}

log_event(event)
print("Drift detection completed")
