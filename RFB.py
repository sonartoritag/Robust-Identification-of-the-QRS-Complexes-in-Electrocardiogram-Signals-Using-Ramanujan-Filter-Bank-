import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from math import gcd


def RFB(samf, xx, Pmax, Rcq, Rav, Th, sigma=100):
    x = np.array(xx, dtype=float).flatten()

    # --- Build Ramanujan filter bank ---
    FR, FA = [], []
    for i in range(1, Pmax + 1):
        cq = np.zeros(i)
        k_vals = [k for k in range(1, i + 1) if gcd(k, i) == 1]
        for n in range(i):
            cq[n] = np.sum(np.cos(2 * np.pi * np.array(k_vals) * n / i))
        cq = cq / np.linalg.norm(cq)
        FR.append(np.tile(cq, Rcq))
        FA.append(np.ones(i * Rav) / np.linalg.norm(np.ones(i * Rav)))

    # --- Compute filter outputs ---
    y = np.zeros((len(x), Pmax))
    for i in range(Pmax):
        y_temp = np.convolve(x, FR[i], mode='same')
        y_temp = np.square(np.abs(y_temp))
        y_temp = np.convolve(y_temp, FA[i], mode='same')
        y[:, i] = y_temp

    # --- Post-processing ---
    y[:, 0] = 0
    y = y - np.min(y)
    y[y < Th] = 0
    y = y.T  # shape: (Pmax, len(x))

    # --- Gaussian smoothing ---
    X = np.sum(y, axis=0)  # sum across periods
    X_smooth = gaussian_filter1d(X, sigma=sigma)  # preserves length

    # --- Peak detection ---
    distance = round(samf * 0.25)
    peaks, _ = find_peaks(X_smooth, distance=distance)
    Rpeaks = peaks.copy()

    # --- Adjust R-peak positions locally ---
    for i in range(len(Rpeaks)):
        smpl = round(samf * 0.02)
        left = max(Rpeaks[i] - smpl, 0)
        right = min(Rpeaks[i] + smpl, len(xx))
        temp = xx[left:right]
        Rpeaks[i] = left + np.argmax(np.abs(temp))

    for i in range(len(Rpeaks)):
        if xx[Rpeaks[i]] <= 0:
            R = Rpeaks[i] - 1
            while R > 1:
                if xx[R] > xx[R - 1] and xx[R] > xx[R + 1]:
                    Rpeaks[i] = R
                    break
                R -= 1

    return Rpeaks, X_smooth
