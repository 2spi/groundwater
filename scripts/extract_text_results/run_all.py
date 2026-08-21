#!/usr/bin/env python
"""Extract every textual result from notebooks/bow_valley/ into outputs/text_results/.

Run with the project environment:

    /home/py/mambaforge/envs/research-gwmo/bin/python scripts/extract_text_results/run_all.py

Takes about a minute end to end; the wavelet step dominates.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import common as c
import n00_inventory, n01_xcorr, n02_hysteresis, n03_trend
import n04_changepoint, n05_shape, n06_wavelet, n07_lithology
import digests, readme, verify

STEPS = [
    ("00 inventory", lambda s: n00_inventory.run(s)),
    ("01 cross-correlation", lambda s: n01_xcorr.run(s)),
    ("02 hysteresis", lambda s: n02_hysteresis.run(s)),
    ("03 trend", lambda s: n03_trend.run(s)),
    ("04 change point", lambda s: n04_changepoint.run(s)),
    ("05 shape", lambda s: n05_shape.run(s)),
    ("06 wavelet", lambda s: n06_wavelet.run(s)),
    ("07 lithology", lambda s: n07_lithology.run()),
]


def main(only=None):
    t0 = time.time()
    print(f"writing to {c.OUT}")
    series = c.load_all()
    print(f"loaded {len(series)} series")

    for name, fn in STEPS:
        if only and not name.startswith(only):
            continue
        t = time.time()
        print(f"{name} ...")
        fn(series)
        print(f"  ({time.time() - t:.1f} s)")

    print("digests ...")
    digests.run()
    print("readme ...")
    readme.run()
    print("verify ...")
    passed = verify.run(series)

    n = sum(1 for p in c.OUT.rglob("*") if p.is_file())
    print(f"\ndone in {time.time() - t0:.0f} s -- {n} files under {c.OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
