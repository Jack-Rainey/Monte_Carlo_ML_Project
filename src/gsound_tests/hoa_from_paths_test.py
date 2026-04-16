import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import pygsound as ps
import spherical_harmonics_rt as sh

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# --- Build the same simple shoebox scene ---
mesh = ps.createbox(10.0, 6.0, 3.0, 0.5, 0.1)

ctx = ps.Context()
ctx.diffuse_count = 20000
ctx.specular_count = 2000
ctx.threads_count = min(multiprocessing.cpu_count(), 8)
ctx.channel_type = ps.ChannelLayoutType.mono
ctx.sample_rate = 16000

scene = ps.Scene()
scene.setMesh(mesh)

src = ps.Source([1.0, 1.0, 1.5])
src.radius = 0.01
src.power = 1.0

lis = ps.Listener([7.0, 4.0, 1.5])
lis.radius = 0.01

# --- Get raw path data ---
paths = scene.getPathData([src], [lis], ctx)["path_data"][0]

num_paths = paths["num_paths"]
num_bands = paths["num_bands"]

print("num_paths:", num_paths)
print("num_bands:", num_bands)

assert num_paths > 0
assert num_bands > 1  # needed because frequency_points must be num_bands - 1

# --- Save a Parquet copy too, for inspection ---
df = pd.DataFrame({
    "listener_direction_x": paths["listener_directions"][:, 0],
    "listener_direction_y": paths["listener_directions"][:, 1],
    "listener_direction_z": paths["listener_directions"][:, 2],
    "distance": paths["distances"],
    "speed_of_sound": paths["speeds_of_sound"],
})

for i in range(num_bands):
    df[f"intensity_band_{i}"] = paths["intensities"][:, i]

parquet_path = output_dir / "shoebox_paths.parquet"
df.to_parquet(parquet_path, index=False)
print(f"Saved {parquet_path}")

# --- Prepare arrays for the auralizer binding ---
listener_directions = paths["listener_directions"].astype(np.float32)
intensities = paths["intensities"].astype(np.float32)
distances = paths["distances"].astype(np.float32)
speeds = paths["speeds_of_sound"].astype(np.float32)

# Binding requires len(frequency_points) == num_bands - 1
# Start with a standard octave-ish split for 8 bands -> 7 crossover points
frequency_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000], dtype=np.float32)

assert len(frequency_points) == num_bands - 1, (
    f"Expected {num_bands - 1} frequency points, got {len(frequency_points)}"
)

# --- Generate a 3rd-order HOA IR ---
hoa_ir = sh.generate_ambisonic_ir(
    3,                      # order
    listener_directions,
    intensities,
    distances,
    speeds,
    frequency_points,
    48000.0,                # output sample rate
    False,                  # precise_early_reflections
    True,                   # normalize
    0.01                    # early_reflection_threshold
)

hoa_ir = np.array(hoa_ir)
print("HOA IR shape:", hoa_ir.shape)
print("HOA IR dtype:", hoa_ir.dtype)

npy_path = output_dir / "shoebox_hoa_ir.npy"
np.save(npy_path, hoa_ir)
print(f"Saved {npy_path}")

# Small sanity checks
assert hoa_ir.ndim == 2, f"Expected 2D HOA IR array, got shape {hoa_ir.shape}"
assert hoa_ir.shape[0] == (3 + 1) ** 2, f"Expected 16 HOA channels, got {hoa_ir.shape[0]}"
assert np.max(np.abs(hoa_ir)) > 0, "Generated HOA IR is all zeros"

print("HOA synthesis test passed.")