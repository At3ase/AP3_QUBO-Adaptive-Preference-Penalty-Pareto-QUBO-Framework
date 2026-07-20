# -*- coding: utf-8 -*-
"""exp0 正式跑批进度一键查看。

用法：双击本文件，或命令行 python scripts/check_progress.py
总轮次 = 20 reps × 5 配置 = 100 轮 ParetoZoom（每轮结束打一行 CONVEXITY）。
"""
import os
import re
import subprocess
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(BASE, "data", "results", "exp0_bg_run_reps20_v2.log")
PID_FILE = os.path.join(BASE, "data", "results", "exp0_bg.pid")
OUT_DIR = os.path.join(BASE, "data", "results", "formal_exp0_reps20", "exp0")
TOTAL_RUNS = 100  # 20 reps × 5 配置

def main():
    print("=" * 56)
    print("AP3-QUBO exp0 正式跑批 · 进度查看")
    print("=" * 56)

    # 1. 进程状态
    pid = None
    if os.path.exists(PID_FILE):
        pid = open(PID_FILE).read().strip()
    alive = False
    if pid:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True, encoding="gbk", errors="ignore",
        ).stdout
        alive = "python" in out.lower()
    print(f"\n[进程] PID={pid}  {'✅ 运行中' if alive else '⛔ 已结束（跑完或异常中断，看日志尾部判断）'}")

    # 2. 日志统计
    if not os.path.exists(LOG):
        print(f"\n[日志] 未找到 {LOG}")
        return
    with open(LOG, encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
    done = sum(1 for ln in lines if "CONVEXITY" in ln)
    pct = done / TOTAL_RUNS * 100
    bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
    print(f"\n[进度] {done}/{TOTAL_RUNS} 轮  |{bar}| {pct:.1f}%")

    # 3. 用时与 ETA
    start = os.path.getmtime(PID_FILE) if os.path.exists(PID_FILE) else None
    if start and done > 0:
        elapsed = time.time() - start
        per_run = elapsed / done
        eta = per_run * (TOTAL_RUNS - done)
        print(f"[用时] 已跑 {elapsed/3600:.1f} 小时，平均每轮 {per_run/60:.1f} 分钟")
        print(f"[预计] 剩余约 {eta/3600:.1f} 小时（粗略估计）")

    # 4. 产物
    if os.path.isdir(OUT_DIR):
        files = os.listdir(OUT_DIR)
    else:
        files = []
    print(f"\n[产物] {OUT_DIR}")
    if files:
        for fn in files:
            print(f"       - {fn}")
    else:
        print("       （空：结果在 20 次重复全部跑完后一次性写入）")

    # 5. 日志最后 3 条实质消息（过滤重复告警）
    print("\n[日志尾部]")
    substantive = [
        ln.rstrip() for ln in lines
        if "D-06" not in ln and "license" not in ln and ln.strip()
    ]
    for ln in substantive[-3:]:
        print(f"       {ln}")

if __name__ == "__main__":
    main()
    if sys.stdin.isatty():
        try:
            input("\n按回车退出...")
        except EOFError:
            pass
