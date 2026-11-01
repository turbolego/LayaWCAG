#!/usr/bin/env python3
"""Download trained WCAG model artifacts from the Kaggle kernel and validate.

Usage: python3 download_artifacts.py
"""
import subprocess, os, json, sys, re

KAGGLE = "/home/kloa/repos/LayaWCAG/kaggle_training/venv/bin/kaggle"
SLUG = "hummern/layawcag-improved-training"
OUTDIR = "/home/kloa/repos/LayaWCAG/kaggle_training/outputs"
os.makedirs(OUTDIR, exist_ok=True)

env = {**os.environ,
       "KAGGLE_USERNAME": "hummern",
       "KAGGLE_KEY": json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["key"]}

# 1. Status
r = subprocess.run([KAGGLE, "kernels", "status", SLUG], capture_output=True, text=True, env=env)
status_line = r.stdout.strip()
print("Status:", status_line)
if "COMPLETE" not in status_line:
    print("Kernel not complete yet — aborting download.")
    sys.exit(1)

# 2. List output files
r = subprocess.run([KAGGLE, "kernels", "output", SLUG], capture_output=True, text=True, env=env)
print("Output files:")
print(r.stdout)

# 3. Download output files
r = subprocess.run([KAGGLE, "kernels", "output", SLUG, "-p", OUTDIR],
                   capture_output=True, text=True, env=env)
print("Download:", r.stdout.strip(), r.stderr.strip())

# 4. Show what landed
for root, _, files in os.walk(OUTDIR):
    for fn in files:
        p = os.path.join(root, fn)
        print(f"  {p}  ({os.path.getsize(p)} bytes)")
