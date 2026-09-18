"""
radar_dsp.py
------------
Core signal-processing chain for an FMCW (Frequency-Modulated Continuous-Wave)
radar simulation: waveform generation, target-echo simulation, dechirping,
Range-FFT, Doppler-FFT (Range-Doppler map), and a CFAR (Constant False Alarm
Rate) target detector.

This module is self-contained (NumPy only) and is written to mirror the
processing chain used in real FMCW radar systems (e.g. automotive radar,
short-range surveillance radar), making it a relevant demonstrator for
radar / EW (Electronic Warfare) signal-processing roles.

Author: Nitish Bugalia
"""

import numpy as np


# ----------------------------------------------------------------------
# 1. RADAR / WAVEFORM CONFIGURATION
# ----------------------------------------------------------------------
class RadarConfig:
    """Holds all FMCW radar system parameters."""

    def __init__(self,
                 f_start=24.0e9,      # chirp start frequency (Hz)  -> 24 GHz ISM band
                 bandwidth=250e6,     # sweep bandwidth (Hz)
                 chirp_time=40e-6,    # duration of one chirp (s)
                 num_chirps=128,      # chirps per frame (slow-time samples)
                 fs=10e6,             # ADC sample rate on the beat signal (Hz)
                 max_range=100.0,     # design max unambiguous range (m)
                 c=3e8):              # speed of light (m/s)
        self.f_start = f_start
        self.bandwidth = bandwidth
        self.chirp_time = chirp_time
        self.num_chirps = num_chirps
        self.fs = fs
        self.max_range = max_range
        self.c = c

        self.slope = bandwidth / chirp_time                # Hz/s
        self.samples_per_chirp = int(fs * chirp_time)
        self.t_fast = np.arange(self.samples_per_chirp) / fs   # fast-time axis
        self.range_resolution = c / (2 * bandwidth)
        self.max_beat_freq = 2 * max_range * self.slope / c
        self.wavelength = c / f_start


# ----------------------------------------------------------------------
# 2. TARGET MODEL
# ----------------------------------------------------------------------
class Target:
    """A single point target with range (m) and radial velocity (m/s)."""

    def __init__(self, range_m, velocity_mps, rcs=1.0):
        self.range_m = range_m
        self.velocity_mps = velocity_mps
        self.rcs = rcs  # relative radar cross-section (amplitude scaling)


# ----------------------------------------------------------------------
# 3. WAVEFORM / ECHO SIMULATION
# ----------------------------------------------------------------------
def simulate_beat_signal(cfg: RadarConfig, targets, snr_db=15.0, seed=42):
    """
    Simulates the dechirped ("beat") signal matrix for a frame of chirps.

    For an FMCW radar, mixing the transmitted chirp with the received
    (delayed + Doppler-shifted) echo produces a beat tone whose frequency
    is proportional to target range, and whose phase changes chirp-to-chirp
    in proportion to target radial velocity.

    Returns
    -------
    beat_matrix : ndarray, shape (num_chirps, samples_per_chirp)
        Complex baseband beat signal for each chirp (rows = slow-time).
    """
    rng = np.random.default_rng(seed)
    n_chirps = cfg.num_chirps
    n_samp = cfg.samples_per_chirp
    beat_matrix = np.zeros((n_chirps, n_samp), dtype=np.complex128)

    for m in range(n_chirps):
        chirp_start_time = m * cfg.chirp_time
        signal = np.zeros(n_samp, dtype=np.complex128)

        for tgt in targets:
            # Round-trip delay updates slowly with target motion (slow-time)
            instantaneous_range = tgt.range_m + tgt.velocity_mps * chirp_start_time
            tau = 2 * instantaneous_range / cfg.c

            # Beat frequency from range (standard FMCW dechirp result)
            f_beat = cfg.slope * tau

            # Dechirped tone: amplitude ~ RCS, phase carries range + Doppler info.
            # NOTE: tau itself already depends on chirp_start_time through target
            # motion (instantaneous_range), so the phase0 term below inherently
            # encodes the Doppler (slow-time) phase progression -- it must NOT be
            # added again as a separate exp(j*2*pi*f_doppler*t) factor, or the
            # velocity estimate would be double-counted.
            phase0 = 2 * np.pi * (cfg.f_start * tau)  # range- and motion-dependent phase
            tone = tgt.rcs * np.exp(1j * (2 * np.pi * f_beat * cfg.t_fast + phase0))
            signal += tone

        beat_matrix[m, :] = signal

    # Add complex AWGN at the requested SNR (relative to a unit-RCS target)
    sig_power = 1.0
    snr_linear = 10 ** (snr_db / 10)
    noise_power = sig_power / snr_linear
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(beat_matrix.shape)
                                         + 1j * rng.standard_normal(beat_matrix.shape))
    return beat_matrix + noise


# ----------------------------------------------------------------------
# 4. RANGE PROCESSING (FAST-TIME FFT)
# ----------------------------------------------------------------------
def range_fft(beat_matrix, cfg: RadarConfig, window=True, n_fft=None):
    """
    Applies a windowed FFT across fast-time (per-chirp) to resolve range.

    Returns
    -------
    range_profile : ndarray, shape (num_chirps, n_fft//2)
        Complex range spectrum for each chirp.
    range_axis : ndarray
        Range (m) corresponding to each FFT bin.
    """
    n_samp = beat_matrix.shape[1]
    n_fft = n_fft or n_samp
    win = np.hanning(n_samp) if window else np.ones(n_samp)
    windowed = beat_matrix * win[np.newaxis, :]

    spectrum = np.fft.fft(windowed, n=n_fft, axis=1)
    half = n_fft // 2
    range_profile = spectrum[:, :half]

    freq_axis = np.fft.fftfreq(n_fft, d=1 / cfg.fs)[:half]
    range_axis = freq_axis * cfg.c / (2 * cfg.slope)
    return range_profile, range_axis


# ----------------------------------------------------------------------
# 5. DOPPLER PROCESSING (SLOW-TIME FFT) -> RANGE-DOPPLER MAP
# ----------------------------------------------------------------------
def doppler_fft(range_profile, cfg: RadarConfig, window=True):
    """
    Applies an FFT across slow-time (chirp index) at every range bin to
    resolve radial velocity, producing the classic Range-Doppler map.

    Returns
    -------
    rd_map_db : ndarray, shape (num_chirps, num_range_bins)
        Range-Doppler map magnitude in dB (zero-Doppler centered).
    velocity_axis : ndarray
    """
    n_chirps, n_range_bins = range_profile.shape
    win = np.hanning(n_chirps) if window else np.ones(n_chirps)
    windowed = range_profile * win[:, np.newaxis]

    rd = np.fft.fftshift(np.fft.fft(windowed, axis=0), axes=0)
    rd_db = 20 * np.log10(np.abs(rd) + 1e-9)

    prf = 1 / cfg.chirp_time  # chirp (pulse) repetition frequency
    doppler_freq = np.fft.fftshift(np.fft.fftfreq(n_chirps, d=1 / prf))
    velocity_axis = doppler_freq * cfg.wavelength / 2
    return rd_db, velocity_axis


# ----------------------------------------------------------------------
# 6. CFAR TARGET DETECTION (1-D Cell-Averaging CFAR on the range profile)
# ----------------------------------------------------------------------
def ca_cfar_1d(power_profile, num_guard=4, num_train=12, pfa=1e-4):
    """
    Cell-Averaging Constant False Alarm Rate (CA-CFAR) detector.

    For every cell under test (CUT), the noise level is estimated from the
    average power of the surrounding *training* cells (excluding a *guard*
    band immediately around the CUT to avoid target energy leaking into the
    noise estimate). A CUT is declared a detection if it exceeds the local
    noise estimate multiplied by a threshold factor (alpha), set from the
    desired probability of false alarm (Pfa).

    Parameters
    ----------
    power_profile : 1D ndarray (linear power, not dB)
    num_guard : int   -- guard cells on each side of the CUT
    num_train : int   -- training cells on each side of the CUT
    pfa : float       -- desired probability of false alarm

    Returns
    -------
    detections : boolean ndarray, same shape as power_profile
    threshold  : ndarray, the adaptive threshold used at every cell
    """
    n = len(power_profile)
    detections = np.zeros(n, dtype=bool)
    threshold = np.zeros(n)

    n_cells = 2 * num_train
    alpha = n_cells * (pfa ** (-1 / n_cells) - 1)  # CA-CFAR threshold factor

    for i in range(num_train + num_guard, n - num_train - num_guard):
        lead_train = power_profile[i - num_train - num_guard: i - num_guard]
        lag_train = power_profile[i + num_guard + 1: i + num_guard + num_train + 1]
        noise_level = np.mean(np.concatenate([lead_train, lag_train]))
        thresh = alpha * noise_level
        threshold[i] = thresh
        if power_profile[i] > thresh:
            detections[i] = True

    return detections, threshold
