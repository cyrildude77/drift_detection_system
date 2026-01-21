
import numpy as np
from scipy.stats import ks_2samp

def ks_test(ref, cur):
    return ks_2samp(ref.flatten(), cur.flatten()).statistic

def psi_test(ref, cur, bins=10):
    ref_hist, _ = np.histogram(ref, bins=bins)
    cur_hist, _ = np.histogram(cur, bins=bins)
    ref_perc = ref_hist / len(ref)
    cur_perc = cur_hist / len(cur)
    psi = np.sum((cur_perc - ref_perc) * np.log((cur_perc + 1e-6) / (ref_perc + 1e-6)))
    return psi
