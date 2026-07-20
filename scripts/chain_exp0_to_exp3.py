# -*- coding: utf-8 -*-
"""Monitor exp0 PID 26104; when it exits, auto-launch experiment 3 (20 reps).

Run this in background — it will block until exp0 finishes, then spawn exp3.
"""
import subprocess
import os
import sys
import time
import ctypes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

PYTHON = r"C:\Users\At3ase\AppData\Local\Programs\Python\Python310\python.exe"
MONITOR_PID = 26104
CHECK_INTERVAL = 30  # seconds

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

SYNCHRONIZE = 0x00100000
_kernel32 = ctypes.windll.kernel32


def pid_alive(pid: int) -> bool:
    """Check if a Windows process exists (doesn't require kill signal)."""
    handle = _kernel32.OpenProcess(SYNCHRONIZE, False, pid)
    if handle == 0:
        return False
    _kernel32.CloseHandle(handle)
    # A non-zero handle means the process exists, but it could be a zombie.
    # Also check exit code.
    h2 = _kernel32.OpenProcess(0x0400, False, pid)  # PROCESS_QUERY_INFORMATION
    if h2 == 0:
        return False
    exit_code = ctypes.c_ulong(0)
    _kernel32.GetExitCodeProcess(h2, ctypes.byref(exit_code))
    _kernel32.CloseHandle(h2)
    return exit_code.value == 259  # STILL_ACTIVE


def main():
    print(f"[chain] 开始监控 PID {MONITOR_PID}，每 {CHECK_INTERVAL}s 检查一次...")
    checks = 0
    while pid_alive(MONITOR_PID):
        checks += 1
        print(f"[chain] 检查 #{checks}: 实验 0 仍在运行 (PID {MONITOR_PID} alive)")
        time.sleep(CHECK_INTERVAL)

    print(f"[chain] 实验 0 (PID {MONITOR_PID}) 已结束，准备启动实验 3 ...")

    log_path = os.path.join(BASE_DIR, "data", "results", "exp3_bg_run_reps20.log")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cmd = [
        PYTHON,
        os.path.join(BASE_DIR, "scripts", "run_experiments.py"),
        "--experiment", "3",
        "--reps", "20",
        "--seed", "42",
        "--out", "data/results/formal_exp3_reps20",
    ]

    print(f"[chain] 启动命令: {' '.join(cmd)}")
    log = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
        close_fds=True,
    )
    pid_file = os.path.join(BASE_DIR, "data", "results", "exp3_bg.pid")
    with open(pid_file, "w") as f:
        f.write(str(proc.pid))
    print(f"[chain] 实验 3 已启动，PID: {proc.pid}")
    print(f"[chain] 输出日志: {log_path}")
    print(f"[chain] 监控脚本退出。")


if __name__ == "__main__":
    main()
