# -*- coding: utf-8 -*-
"""查看连续运行日志的进度。

用法：双击同目录下的「查看进度.bat」。

这个脚本只读、不写任何文件，随时可以跑，不会污染日志。
"""
import json
import re
import subprocess
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "process_logs"
TASK_NAME = "ICEC_DailyInference"

TASK_LABEL = {"form": "形态", "light": "光照", "amount": "数量"}
LINE = "=" * 64


def dwidth(s):
    """按终端显示宽度计算字符串长度（中文占 2 列）。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s, w, right=False):
    d = max(0, w - dwidth(s))
    return (" " * d + s) if right else (s + " " * d)


def parse_ts(s):
    """兼容带毫秒与不带毫秒两种 ISO 时间戳。"""
    s = s.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")


def query_task_ps():
    """schtasks 不可用时的回退方案。"""
    ps = (
        '$i = Get-ScheduledTaskInfo -TaskName "%s" -ErrorAction Stop; '
        'Write-Output ("NEXT=" + $i.NextRunTime.ToString("yyyy-MM-dd HH:mm:ss") '
        '+ "|LAST=" + $i.LastRunTime.ToString("yyyy-MM-dd HH:mm:ss") '
        '+ "|RC=" + $i.LastTaskResult)'
    ) % TASK_NAME
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=30,
        )
    except Exception:
        return None
    out = proc.stdout.decode("gbk", errors="replace")
    m = re.search(r"NEXT=([^|]*)\|LAST=([^|]*)\|RC=([-\d]+)", out)
    if not m:
        return None
    return {"next": m.group(1), "last": m.group(2), "result": m.group(3)}


def query_task():
    """读取计划任务的运行状态。返回 (info, 原因)。"""
    try:
        proc = subprocess.run(
            ["schtasks", "/query", "/tn", TASK_NAME, "/fo", "LIST", "/v"],
            capture_output=True, timeout=15,
        )
        if proc.returncode == 0:
            out = proc.stdout.decode("gbk", errors="replace")
            info = {}
            for line in out.splitlines():
                low = line.strip()
                for key, tag in (("下次", "next"), ("上次运行", "last"), ("上次结果", "result")):
                    if low.startswith(key) and tag not in info:
                        info[tag] = low.split(":", 1)[-1].strip()
            if info:
                return info, ""
    except Exception:
        pass

    # schtasks 被拦或不可用，改用 PowerShell 读一次
    ps_info = query_task_ps()
    if ps_info:
        return ps_info, ""
    return None, "schtasks 与 PowerShell 都读不到（可能被安全软件限制）"


def main():
    print(LINE)
    print("  ICEC 项目二 · 连续运行检测日志 · 进度查看")
    print("  查询时间 " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(LINE)

    files = sorted(LOG_DIR.glob("*.jsonl")) if LOG_DIR.exists() else []
    if not files:
        print("\n  process_logs 里还没有任何 .jsonl 日志。")
        print("  刚配好定时任务的话，等第一次触发（每天 21:00）后就会出现。\n")
        print(LINE)
        return

    # ---------- 逐日文件 ----------
    print("\n【日志文件】")
    days = []
    for f in files:
        rows = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
        if not rows:
            continue
        ts = sorted(parse_ts(r["timestamp"]) for r in rows)
        span = (ts[-1] - ts[0]).total_seconds()
        days.append((f.stem, rows, ts, span))
        print("  %s %s  %s ~ %s  （%d 秒跑完）"
              % (pad(f.stem, 13), pad("%d 条" % len(rows), 9, True),
                 ts[0].strftime("%H:%M:%S"), ts[-1].strftime("%H:%M:%S"), span))

    total = sum(len(d[1]) for d in days)
    print("  " + "-" * 58)
    print("  合计 %d 天 / %d 条" % (len(days), total))

    # ---------- 逐日准确率 ----------
    tasks = sorted({r["task"] for d in days for r in d[1]})
    print("\n【每天各任务准确率】")
    print("  " + pad("日期", 13) + pad("条数", 7, True)
          + "".join(pad(TASK_LABEL.get(t, t) + "分类", 12, True) for t in tasks))
    for name, rows, _, _ in days:
        cells = ""
        for t in tasks:
            sub = [r for r in rows if r["task"] == t]
            ok = sum(1 for r in sub if not r.get("is_mismatch"))
            cells += pad("%.1f%%" % (100 * ok / len(sub)) if sub else "-", 12, True)
        print("  " + pad(name, 13) + pad(str(len(rows)), 7, True) + cells)

    # ---------- 累计 ----------
    agg = defaultdict(lambda: [0, 0])
    for d in days:
        for r in d[1]:
            agg[r["task"]][1] += 1
            if not r.get("is_mismatch"):
                agg[r["task"]][0] += 1
    print("\n【两日以上累计】")
    for t in tasks:
        ok, n = agg[t]
        print("  " + pad(TASK_LABEL.get(t, t) + "分类", 12) + "%d / %d = %.1f%%" % (ok, n, 100 * ok / n))

    # ---------- 定时任务 ----------
    info, reason = query_task()
    print("\n【定时任务 %s】" % TASK_NAME)
    if info:
        print("  下次运行  " + info.get("next", "(未知)"))
        print("  上次运行  " + info.get("last", "(未知)"))
        print("  返回码    " + info.get("result", "(未知)") + "   （0 = 成功）")
    else:
        print("  读取失败：" + reason)
        print("  可在「任务计划程序」（Win+R 输入 taskschd.msc）里手动查看。")

    # ---------- 结论 ----------
    print("\n【为什么看不到它在跑】")
    print("  它不是常驻进程，是每天 21:00 被系统唤起一次、几十秒跑完就退出。")
    print("  所以任务管理器里看不到，这是正常的。")
    print("  验证方式是看日期列表：每跑一天多一行，一行 60 条。")
    print("  明天这个时间再双击本文件，如果多出一行，就说明自动运行生效了。")

    print("\n" + LINE)
    print("  更详细的汇总见 process_logs\\summary.md")
    print(LINE)


if __name__ == "__main__":
    main()
