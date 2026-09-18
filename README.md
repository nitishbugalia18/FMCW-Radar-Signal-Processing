# FMCW Radar Signal Processing — Range-Doppler Estimation & CFAR Target Detection

A from-scratch Python (NumPy) implementation of the core digital signal-processing
chain used in real FMCW (Frequency-Modulated Continuous-Wave) radar systems —
the same class of processing used in automotive radar, short-range surveillance
radar, and drone-detection radar.

## What it does

1. **Waveform & echo simulation** — generates a linear FM chirp, simulates one or
   more moving point targets (range + radial velocity + RCS), and produces the
   dechirped ("beat") signal with additive noise at a configurable SNR.
2. **Range processing** — windowed FFT across fast-time to resolve target range
   from the beat frequency.
3. **Doppler processing** — a second FFT across slow-time (chirp index) to
   resolve radial velocity, producing the classic **Range-Doppler map**.
4. **CFAR detection** — a Cell-Averaging CFAR (CA-CFAR) detector that
   automatically declares targets from the noisy range profile at a
   configurable probability of false alarm (Pfa), without a fixed threshold.

## Why this project

Radar, EW (electronic warfare), and anti-drone systems all depend on exactly
this pipeline — chirp generation, matched filtering / range-FFT, Doppler
processing, and CFAR detection — to turn raw ADC samples into target reports.
This project reproduces that pipeline end-to-end in software, including a
realistic CFAR "masking" limitation near strong targets, to demonstrate
practical understanding of radar/DSP fundamentals (not just textbook FFT use).

## Project structure

```
radar_project/
├── radar_dsp.py     # Core DSP library: waveform, echo sim, range-FFT, doppler-FFT, CFAR
├── main.py          # End-to-end demo: runs the chain and saves all figures
├── requirements.txt
├── outputs/         # Generated figures (created by running main.py)
│   ├── 1_beat_signal.png
│   ├── 2_range_profile.png
│   ├── 3_range_doppler_map.png
│   └── 4_cfar_detection.png
└── README.md
```

## Run it

```bash
pip install -r requirements.txt
python3 main.py
```

This simulates two targets (30 m / +8 m/s and 65 m / -3 m/s), prints the
CFAR-detected ranges, and saves all four figures to `outputs/`.

## Sample results

| Stage | Output |
|---|---|
| Range-FFT | Correctly resolves both targets at 30 m and 65 m |
| Doppler-FFT (Range-Doppler map) | Correctly resolves +8 m/s and -3 m/s radial velocity |
| CA-CFAR | Detects both targets at Pfa = 1e-4; shows realistic threshold "masking" near strong returns |

## Possible extensions

- 2-D CFAR directly on the Range-Doppler map
- Ordered-Statistic CFAR (OS-CFAR) to reduce masking in multi-target scenes
- MIMO / angle-of-arrival (beamforming) processing for a 3-D (range-doppler-angle) cube
- Replacing the simulated beat signal with real SDR-captured I/Q data

## Author

Nitish Bugalia — B.Tech ECE, VIT Chennai
