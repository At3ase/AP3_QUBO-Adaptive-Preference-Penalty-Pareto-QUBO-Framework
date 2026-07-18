# -*- coding: utf-8 -*-
"""Detach-spawn exp0 formal run so it survives the Bash tool's 300s foreground cap."""
import subprocess
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
os.chdir(BASE_DIR)
log_path = os.path.join(BASE_DIR, "data", "results", "exp0_bg_run.log")
os.makedirs(os.path.dirname(log_path), exist_ok=True)

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

cmd = [
    r"C:\Users\At3ase\AppData\Local\Programs\Python\Python310\python.exe",
    os.path.join(BASE_DIR, "scripts", "run_experiments.py"),
    "--experiment", "0",
    "--reps", "1",
    "--seed", "42",
    "--out", "data/results/formal_exp0_r1",
]

log = open(log_path, "ab", buffering=0)
proc = subprocess.Popen(
    cmd,
    stdout=log,
    stderr=subprocess.STDOUT,
    stdin=subprocess.DEVNULL,
    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    close_fds=True,
)
with open(os.path.join(BASE_DIR, "data", "results", "exp0_bg.pid"), "w") as f:
    f.write(str(proc.pid))
print("PID:", proc.pid)
