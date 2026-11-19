#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""扫描 process_logs/*.jsonl，按日期聚合，生成 process_logs/summary.md。

输出内容：
  1. 按「日期 × 任务」排列的明细表：张数 / 平均预处理耗时 / 平均推理耗时 /
     平均置信度 / 误报数 / 误报率
  2. 按真实类别统计的误报分布（哪一类最容易被错判）
  3. 按场景统计的误报率（对应「抗光照干扰」这个题目）
  4. 一句趋势结论：置信度有没有随天数变化，耗时有没有波动

所有数字都从 jsonl 现场算出，不读取任何预先写好的指标。

用法：
    python summarize.py
    python summarize.py --dir process_logs --out process_logs/summary.md
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DIR = HERE / "process_logs"

TASK_CN = {"form": "形态三分类", "light": "光照四分类", "amount": "数量二分类"}


def load_rows(log_dir: Path):
    """读取目录下全部 jsonl，返回记录列表（附带日期）。"""
    rows = []
    for p in sorted(log_dir.glob("*.jsonl")):
        try:
            day = date.fromisoformat(p.stem)
        except ValueError:
            continue
        for ln, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                print("  [warn] 跳过无法解析的行：%s:%d" % (p.name, ln))
                continue
            r["_date"] = day
            rows.append(r)
    return rows


def mean(xs):
    return statistics.fmean(xs) if xs else 0.0


def fmt(x, nd=1):
    return ("%." + str(nd) + "f") % x


def build(rows):
    """生成 summary.md 的文本。"""
    if not rows:
        return "# 连续运行检测日志 · 汇总\n\n还没有任何 jsonl 记录。先运行 `daily_inference.py`。\n"

    days = sorted(set(r["_date"] for r in rows))
    tasks = sorted(set(r.get("task", "?") for r in rows))

    out = []
    out.append("# 连续运行检测日志 · 汇总\n")
    out.append("> 本文件由 `summarize.py` 扫描 `process_logs/*.jsonl` 自动生成，"
               "所有数字均可从原始 jsonl 复算。\n")
    out.append("**覆盖范围**　%s ～ %s，连续 %d 天，共 %d 条记录，任务：%s\n"
               % (days[0].isoformat(), days[-1].isoformat(), len(days), len(rows),
                  "、".join("%s（%s）" % (t, TASK_CN.get(t, t)) for t in tasks)))

    # ---------- 1. 按日期 × 任务 ----------
    out.append("\n## 1. 每日本次运行明细\n")
    out.append("| 日期 | 任务 | 张数 | 平均预处理 ms | 平均推理 ms | 平均端到端 ms | 平均置信度 | 误报数 | 误报率 |")
    out.append("|---|---|---|---|---|---|---|---|---|")

    by_day = defaultdict(list)
    for r in rows:
        by_day[(r["_date"], r.get("task", "?"))].append(r)

    for key in sorted(by_day):
        d, t = key
        items = by_day[key]
        n = len(items)
        miss = sum(1 for r in items if r.get("is_mismatch"))
        out.append("| %s | %s | %d | %s | %s | %s | %s | %d | %s%% |" % (
            d.isoformat(), t, n,
            fmt(mean([r.get("preprocess_ms", 0) for r in items])),
            fmt(mean([r.get("inference_ms", 0) for r in items])),
            fmt(mean([r.get("total_ms", 0) for r in items])),
            fmt(mean([r.get("confidence", 0) for r in items]), 3),
            miss, fmt(miss / n * 100),
        ))

    # ---------- 2. 按真实类别 ----------
    out.append("\n## 2. 按真实类别的误判分布\n")
    out.append("误判（`is_mismatch`）定义为：预测类别 ≠ 目录名标签。"
               "真值来自文件夹名，属于**已知真值的离线回放验证**，不是现场盲测。"
               "三个任务的类别空间不同，分开统计。\n")

    for t in tasks:
        t_rows = [r for r in rows if r.get("task") == t]
        if not t_rows:
            continue
        out.append("### 任务 `%s`（%s）\n" % (t, TASK_CN.get(t, t)))
        out.append("| 真实类别 | 张数 | 误判数 | 误判率 | 最常被错判为 |")
        out.append("|---|---|---|---|---|")
        by_label = defaultdict(list)
        for r in t_rows:
            by_label[r.get("label", "?")].append(r)
        for lab in sorted(by_label, key=lambda k: -len(by_label[k])):
            items = by_label[lab]
            bad = [r for r in items if r.get("is_mismatch")]
            top = Counter(r.get("pred") for r in bad).most_common(1)
            out.append("| %s | %d | %d | %s%% | %s |" % (
                lab, len(items), len(bad), fmt(len(bad) / len(items) * 100),
                ("%s（%d 次）" % (top[0][0], top[0][1])) if top else "—",
            ))
        out.append("")

    # ---------- 3. 按场景 ----------
    out.append("\n## 3. 按采集场景的误判率\n")
    out.append("这一节直接对应「抗光照干扰」：哪一组光照/形态最难。"
               "同一场景可能被不同任务各跑一次，因此按任务分开列。\n")

    for t in tasks:
        t_rows = [r for r in rows if r.get("task") == t]
        if not t_rows:
            continue
        out.append("### 任务 `%s`\n" % t)
        out.append("| 场景 | 张数 | 误判数 | 误判率 | 平均置信度 |")
        out.append("|---|---|---|---|---|")
        by_scene = defaultdict(list)
        for r in t_rows:
            by_scene[r.get("scene", "?")].append(r)
        for sc in sorted(by_scene, key=lambda k: -len(by_scene[k])):
            items = by_scene[sc]
            miss = sum(1 for r in items if r.get("is_mismatch"))
            out.append("| %s | %d | %d | %s%% | %s |" % (
                sc, len(items), miss, fmt(miss / len(items) * 100),
                fmt(mean([r.get("confidence", 0) for r in items]), 3),
            ))
        out.append("")

    # ---------- 4. 趋势 ----------
    out.append("\n## 4. 趋势结论\n")
    day_stat = []
    for d in days:
        items = [r for r in rows if r["_date"] == d]
        day_stat.append((d, mean([r.get("confidence", 0) for r in items]),
                         mean([r.get("inference_ms", 0) for r in items]), len(items)))

    if len(day_stat) < 4:
        first, last = day_stat[0], day_stat[-1]
        out.append("- 目前只有 %d 天的记录（%s ～ %s），**样本天数不足，本节暂不下趋势结论**。"
                   "按计划跑到 11-21（共 7 天）后再看这一节。"
                   % (len(day_stat), first[0].isoformat(), last[0].isoformat()))
        out.append("- 供参考的观察值：平均置信度 %.3f → %.3f（%+.3f）；"
                   "平均推理耗时 %.1f → %.1f ms（%+.1f ms）。"
                   % (first[1], last[1], last[1] - first[1],
                      first[2], last[2], last[2] - first[2]))
        out.append("- 需要说明：每日抽样是**不同子集**（按日期做种子轮转），"
                   "因此上述差值主要反映抽样场景构成的差异，不能解读为模型变好或变差。")
        out.append("- 累计 %d 天、%d 条记录，总误判 %d 条（%s%%）。"
                   % (len(days), len(rows), sum(1 for r in rows if r.get("is_mismatch")),
                      fmt(sum(1 for r in rows if r.get("is_mismatch")) / len(rows) * 100)))
    else:
        first, last = day_stat[0], day_stat[-1]
        dc = last[1] - first[1]
        di = last[2] - first[2]
        out.append("- 置信度：第 1 天 %.3f → 最近一天 %.3f，变化 %+.3f。%s"
                   % (first[1], last[1], dc,
                      "基本持平，抽样波动范围内。" if abs(dc) < 0.02 else
                      ("呈上升趋势。" if dc > 0 else "有所下降，需结合当天的场景分布看。")))
        out.append("- 推理耗时：第 1 天 %.1f ms → 最近一天 %.1f ms，变化 %+.1f ms。%s"
                   % (first[2], last[2], di,
                      "稳定。" if abs(di) < 15 else "波动较大，检查是否有其他进程抢占 CPU。"))
        confs = [s[1] for s in day_stat]
        out.append("- 全期置信度区间 %.3f ～ %.3f，标准差 %.4f；单日最大波动 %.3f。"
                   % (min(confs), max(confs), statistics.pstdev(confs), max(confs) - min(confs)))
        infs = [s[2] for s in day_stat]
        out.append("- 全期推理耗时区间 %.1f ～ %.1f ms，极差 %.1f ms。"
                   % (min(infs), max(infs), max(infs) - min(infs)))
        out.append("- 累计 %d 天、%d 条记录，总误判 %d 条（%s%%）。"
                   % (len(days), len(rows), sum(1 for r in rows if r.get("is_mismatch")),
                      fmt(sum(1 for r in rows if r.get("is_mismatch")) / len(rows) * 100)))

    out.append("\n---\n")
    out.append("**说明**：每日抽样 30 张，按日期做种子、按场景分层轮转，"
               "保证覆盖全部 12 个目录（含表现最差的阴影组）。"
               "因此单日误判率的抽样误差约 ±8 个百分点，跨日比较时应看趋势而非单日绝对值。\n")

    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="汇总 jsonl 日志为 summary.md")
    ap.add_argument("--dir", default=None, help="日志目录，默认 process_logs/")
    ap.add_argument("--out", default=None, help="输出文件，默认 <dir>/summary.md")
    args = ap.parse_args()

    log_dir = Path(args.dir) if args.dir else DEFAULT_DIR
    out_path = Path(args.out) if args.out else (log_dir / "summary.md")

    rows = load_rows(log_dir)
    text = build(rows)
    out_path.write_text(text, encoding="utf-8")

    print("读取 %d 条记录，覆盖 %d 天" % (len(rows), len(set(r["_date"] for r in rows))))
    print("已写入", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
