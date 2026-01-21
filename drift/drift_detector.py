
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from stats.statistical_tests import ks_test, psi_test

class DriftDetector:
    def __init__(self, pca_model, lstm_model, threshold=0.1):
        self.pca = pca_model
        self.lstm = lstm_model
        self.threshold = threshold

    def detect_pca_drift(self, X):
        X_recon = self.pca.inverse_transform(self.pca.transform(X))
        error = np.mean((X - X_recon)**2)
        return error

    def detect_lstm_drift(self, X):
        X_pred = self.lstm.predict(X, verbose=0)
        error = np.mean((X - X_pred)**2)
        return error

    def detect_stat_drift(self, ref, cur):
        return ks_test(ref, cur), psi_test(ref, cur)