from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs"

hoa_path = output_dir / "shoebox_hoa_ir.npy"
parquet_path = output_dir / "shoebox_paths.parquet"

hoa_ir = np.load(hoa_path)
df = pd.read_parquet(parquet_path)

print("=== HOA IR ===")
print("Path:", hoa_path)
print("Shape:", hoa_ir.shape)
print("Dtype:", hoa_ir.dtype)

num_channels, num_samples = hoa_ir.shape
sample_rate = 48000  # matches your synthesis script
duration_sec = num_samples / sample_rate

print("Channels:", num_channels)
print("Samples:", num_samples)
print("Duration (s):", duration_sec)
print("Global max abs:", np.max(np.abs(hoa_ir)))

channel_peaks = np.max(np.abs(hoa_ir), axis=1)
print("Peak per channel:", channel_peaks)

print("\n=== Path Data ===")
print("Path:", parquet_path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head())

intensity_cols = [c for c in df.columns if c.startswith("intensity_band_")]
print("\nIntensity columns:", intensity_cols)

print("\nDistance stats:")
print(df["distance"].describe())

print("\nSpeed-of-sound stats:")
print(df["speed_of_sound"].describe())

# Plot first HOA channel
plt.figure(figsize=(10, 4))
plt.plot(hoa_ir[0])
plt.title("HOA IR - Channel 0")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()