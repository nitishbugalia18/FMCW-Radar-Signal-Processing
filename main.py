"""
main.py
-------
End-to-end demonstration of the FMCW radar DSP chain:
  1. Simulate two moving targets (different range + velocity)
  2. Generate the dechirped beat-signal matrix (with AWGN)
  3. Range-FFT  -> range profile per chirp
  4. Doppler-FFT -> Range-Doppler map
  5. CA-CFAR   -> automatic target detection on the range profile
  6. Save all figures to ./outputs for the project report / README

Run:  python3 main.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from radar_dsp import RadarConfig, Target, simulate_beat_signal, range_fft, doppler_fft, ca_cfar_1d

OUT = "outputs"

def main():
    cfg = RadarConfig()
    print(f"Range resolution      : {cfg.range_resolution:.3f} m")
    print(f"Max unambiguous range : {cfg.max_range:.1f} m")
    print(f"Wavelength            : {cfg.wavelength*1e3:.2f} mm")
    print(f"Samples per chirp     : {cfg.samples_per_chirp}")

    # ---- Ground-truth targets -------------------------------------------------
    targets = [
        Target(range_m=30.0, velocity_mps=8.0, rcs=1.0),   # e.g. approaching drone/vehicle
        Target(range_m=65.0, velocity_mps=-3.0, rcs=0.85),  # receding, weaker target
    ]
    print("\nGround truth:")
    for i, t in enumerate(targets):
        print(f"  Target {i+1}: range = {t.range_m} m, velocity = {t.velocity_mps} m/s, RCS = {t.rcs}")

    # ---- 1. Simulate beat signal ------------------------------------------------
    beat_matrix = simulate_beat_signal(cfg, targets, snr_db=12.0)

    plt.figure(figsize=(8, 4))
    plt.plot(cfg.t_fast * 1e6, np.real(beat_matrix[0, :]))
    plt.title("Dechirped Beat Signal — Single Chirp (Real Part)")
    plt.xlabel("Fast Time (µs)")
    plt.ylabel("Amplitude")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/1_beat_signal.png", dpi=150)
    plt.close()

    # ---- 2. Range FFT -> range profile ------------------------------------------
    range_profile, range_axis = range_fft(beat_matrix, cfg)
    single_chirp_profile_db = 20 * np.log10(np.abs(range_profile[0, :]) + 1e-9)

    plt.figure(figsize=(8, 4))
    plt.plot(range_axis, single_chirp_profile_db)
    for t in targets:
        plt.axvline(t.range_m, color="r", linestyle="--", alpha=0.6)
    plt.title("Range Profile (Single Chirp) — Range-FFT Output")
    plt.xlabel("Range (m)")
    plt.ylabel("Magnitude (dB)")
    plt.xlim(0, cfg.max_range)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/2_range_profile.png", dpi=150)
    plt.close()

    # ---- 3. Doppler FFT -> Range-Doppler map ------------------------------------
    rd_map_db, velocity_axis = doppler_fft(range_profile, cfg)

    plt.figure(figsize=(7, 6))
    extent = [range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]]
    plt.imshow(rd_map_db, aspect="auto", origin="lower", extent=extent, cmap="viridis")
    plt.colorbar(label="Magnitude (dB)")
    for t in targets:
        plt.scatter(t.range_m, t.velocity_mps, edgecolors="red", facecolors="none", s=140, linewidths=2)
    plt.title("Range-Doppler Map")
    plt.xlabel("Range (m)")
    plt.ylabel("Radial Velocity (m/s)")
    plt.xlim(0, cfg.max_range)
    plt.tight_layout()
    plt.savefig(f"{OUT}/3_range_doppler_map.png", dpi=150)
    plt.close()

    # ---- 4. CFAR detection on the (noncoherently integrated) range profile -----
    power_profile = np.mean(np.abs(range_profile) ** 2, axis=0)  # integrate across chirps
    detections, threshold = ca_cfar_1d(power_profile, num_guard=4, num_train=12, pfa=1e-4)

    detected_ranges = range_axis[detections]
    print("\nCFAR-detected target ranges (m):", np.round(detected_ranges, 2))

    plt.figure(figsize=(8, 4))
    plt.plot(range_axis, 10 * np.log10(power_profile + 1e-12), label="Integrated Range Profile")
    plt.plot(range_axis, 10 * np.log10(threshold + 1e-12), label="CFAR Adaptive Threshold", linestyle="--")
    plt.scatter(range_axis[detections], 10 * np.log10(power_profile[detections] + 1e-12),
                color="red", zorder=5, label="CFAR Detections")
    for t in targets:
        plt.axvline(t.range_m, color="gray", linestyle=":", alpha=0.7)
    plt.title("CA-CFAR Target Detection")
    plt.xlabel("Range (m)")
    plt.ylabel("Power (dB)")
    plt.xlim(0, cfg.max_range)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/4_cfar_detection.png", dpi=150)
    plt.close()

    print("\nAll figures saved to ./outputs/")


if __name__ == "__main__":
    main()
