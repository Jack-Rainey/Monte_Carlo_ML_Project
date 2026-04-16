from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import numpy as np
import pygsound as ps
import spherical_harmonics_rt as sh


import pandas as pd  # noqa: F401
import pyarrow  # noqa: F401

from scene_gen.dataset_builder import DatasetBuilder  # noqa: F401
from orchestration.dataset_workflow import build_dataset  # noqa: F401


def main():
    length_m = 13.325070568460415
    width_m = 5.147769107417625
    height_m = 3.5714294327697416
    absorption = 0.37606026143308635
    scattering = 0.1

    mesh = ps.createbox(length_m, width_m, height_m, absorption, scattering)

    scene = ps.Scene()
    scene.setMesh(mesh)

    source = ps.Source([6.01191366427458, 3.7987895563432934, 1.486896615752596])
    source.radius = 0.01
    source.power = 1.0

    listener = ps.Listener([1.3464361271329106, 2.96814009105892, 1.5851150009367614])
    listener.radius = 0.01

    ctx = ps.Context()
    ctx.diffuse_count = 9091
    ctx.specular_count = 909
    ctx.threads_count = 1
    ctx.channel_type = ps.ChannelLayoutType.mono
    ctx.sample_rate = 48000

    print("about to call getPathData")
    result = scene.getPathData([source], [listener], ctx)
    print("success")
    print(type(result))
    print(result.keys())


if __name__ == "__main__":
    main()