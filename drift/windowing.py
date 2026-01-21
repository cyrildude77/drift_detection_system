
import numpy as np

def sliding_window(data, window_size=50, stride=10):
    windows = []
    for i in range(0, len(data) - window_size, stride):
        windows.append(data[i:i+window_size])
    return np.array(windows)
