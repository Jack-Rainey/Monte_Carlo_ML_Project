import multiprocessing
import numpy as np
import pygsound as ps
import wave

# Build a simple shoebox room
mesh = ps.createbox(10.0, 6.0, 3.0, 0.5, 0.1)

# Configure the simulation
ctx = ps.Context()
ctx.diffuse_count = 20000
ctx.specular_count = 2000
ctx.threads_count = min(multiprocessing.cpu_count(), 8)
ctx.channel_type = ps.ChannelLayoutType.mono
ctx.sample_rate = 16000

# Create scene
scene = ps.Scene()
scene.setMesh(mesh)

# Source / listener
src = ps.Source([1.0, 1.0, 1.5])
src.radius = 0.01
src.power = 1.0

lis = ps.Listener([7.0, 4.0, 1.5])
lis.radius = 0.01

# Compute IR
res = scene.computeIR([src], [lis], ctx)
ir = np.array(res["samples"][0][0][0], dtype=np.float32)

# Basic sanity checks like the repo's tests
assert np.max(np.abs(ir)) > 0, "IR is all zeros"
assert not np.isnan(ir).any(), "IR contains NaNs"

# Normalize to int16 WAV
peak = np.max(np.abs(ir))
if peak > 0:
    ir = ir / peak

ir_i16 = (ir * 32767).astype(np.int16)

with wave.open("shoebox_ir.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(int(res["rate"]))
    wf.writeframes(ir_i16.tobytes())

print("Saved shoebox_ir.wav")
print("Sample rate:", res["rate"])
print("IR length:", len(ir))
print("Peak sample index:", int(np.argmax(np.abs(ir))))