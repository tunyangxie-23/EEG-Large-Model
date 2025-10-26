"""
Prototype helpers for multi-band FIR decomposition of EEG.
"""
from typing import Iterable, Sequence, Tuple

import mne
import numpy as np


DEFAULT_BANDS: Tuple[Tuple[float, float], ...] = (
    (0.5, 4.0),
    (4.0, 8.0),
    (8.0, 13.0),
    (13.0, 30.0),
    (30.0, 50.0),
)


def apply_fir_bandpass(data: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    """
    Run an FIR band-pass filter on EEG data.
    data: (channels, time)
    """
    return mne.filter.filter_data(
        data.copy(), sfreq=sfreq,
        l_freq=low, h_freq=high,
        method='fir', phase='zero',
        fir_window='hamming', fir_design='firwin',
        verbose=False
    )


def cumulative_frequency_levels(
    data: np.ndarray,
    sfreq: float,
    bands: Sequence[Tuple[float, float]] = DEFAULT_BANDS,
) -> np.ndarray:
    """
    Produce cumulative sum of band-limited signals.
    Returns array shaped (len(bands), channels, time).
    """
    band_signals = [apply_fir_bandpass(data, sfreq, low, high) for low, high in bands]
    stacked = np.stack(band_signals, axis=0)
    return np.cumsum(stacked, axis=0)
