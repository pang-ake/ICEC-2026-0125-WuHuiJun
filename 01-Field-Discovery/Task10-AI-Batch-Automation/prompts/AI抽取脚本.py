# -*- coding: utf-8 -*-
"""
日常数字任务耗时审计 · AI 批处理脚本
================================================
任务：从校园采购公示文本（docx）中批量抽取「日期 / 公司名 / 金额」三个字段，
      逐份计时，并做字段级校验告警。

用法：
    python AI抽取脚本.py                    # 处理同目录 样本1—样本10.docx
    python AI抽取脚本.py --out 结果.csv      # 额外导出 CSV
    python AI抽取脚本.py --dir D:\\某目录     # 指定样本目录

依赖：python-docx（pip install python-docx）
"""

import argparse
import csv
import os
import re
import sys
import time

try:
    from docx import Document
except ImportError:
    sys.exit("缺少依赖，请先执行：pip install python-docx")

# ----------------------------------------------------------------------------
# 方案 A：规则抽取（本脚本实现，毫秒级）
# 方案 B：大模型抽取（Prompt 见《AI_Prompt模板.md》，把下面的 PROMPT 拼接正文发给模型）
# ----------------------------------------------------------------------------
PROMPT = """你是公文信息抽取器。从下面的校园采购文本中抽取三个字段，只输出 JSON，不要解释。

字段定义：
- date    ：采购/招标/实施/立项等主体动作发生的日期，格式 YYYY-MM-DD
- company ：中标、承接、供货或赞助单位的完整名称，不加书名号引号
- amount  ：金额，统一折算为「元」的整数；「万元」需乘 10000

规则：缺失填 null，不要猜测；同一文本出现多个日期时，取与采购动作绑定的那一个。
输出格式：{"date": "...", "company": "...", "amount": 0, "evidence": "原文中的关键短语"}

文本：
{text}"""

# 正则：日期、公司名（引号内以公司/基金会/集团结尾）、金额（数字 + 元/万元）
RE_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
RE_COMPANY = re.compile(r"[“\"]([^”\"]+?(?:公司|基金会|集团|研究院))[”\"]")
RE_AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万元|元)")

UNIT = {"元": 1, "万元": 10000}


def load_text(path):
    """读取 docx 全部段落，返回 (标题, 正文)。"""
    paras = [p.text.strip() for p in Document(path).paragraphs if p.text.strip()]
    if not paras:
        return "", ""
    return paras[0], "\n".join(paras[1:]) if len(paras) > 1 else paras[0]


def extract(text):
    """从正文抽取三个字段，返回记录 dict（含校验告警）。"""
    rec = {"date": None, "company": None, "amount": None, "raw": None, "warn": []}

    # --- 日期：统计候选个数，多于一个时告警（对应 R4）
    dates = list(RE_DATE.finditer(text))
    if dates:
        y, m, d = dates[0].groups()
        rec["date"] = "%s-%02d-%02d" % (y, int(m), int(d))
        if len(dates) > 1:
            rec["warn"].append("R4 日期多候选(%d 处)，已取首个" % len(dates))
    else:
        rec["warn"].append("R1 未匹配到日期")

    # --- 公司名：引号内的主体名（对应 R5/R6）
    m = RE_COMPANY.search(text)
    if m:
        rec["company"] = m.group(1)
        if any(bad in rec["company"] for bad in ("我校", "学院", "中心", "部")):
            if not rec["company"].endswith(("公司", "基金会", "集团", "研究院")):
                rec["warn"].append("R6 主体名疑似非外部单位")
    else:
        rec["warn"].append("R1 未匹配到公司名")

    # --- 金额：统一折算为元（对应 R2）
    a = RE_AMOUNT.search(text)
    if a:
        num, unit = a.groups()
        rec["raw"] = "%s %s" % (num, unit)
        rec["amount"] = int(round(float(num.replace(",", "")) * UNIT[unit]))
        if unit == "万元":
            rec["warn"].append("R2 原文为「万元」，已自动折算为元")
        if rec["amount"] == 0 or rec["amount"] > 10_000_000:
            rec["warn"].append("R3 金额量级异常，建议人工确认")
    else:
        rec["warn"].append("R1 未匹配到金额")

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--out", default=None, help="导出 CSV 的路径")
    args = ap.parse_args()

    files = sorted(
        [f for f in os.listdir(args.dir) if f.endswith(".docx") and f.startswith("样本")],
        key=lambda x: int("".join(c for c in x if c.isdigit()) or 0),
    )
    if not files:
        sys.exit("未找到样本文件（样本*.docx）")

    t_all = time.perf_counter()
    records = []
    print("%-4s %-12s %-14s %-24s %10s %8s" % ("序号", "文件", "日期", "公司名", "金额(元)", "耗时ms"))
    print("-" * 92)
    for i, fname in enumerate(files, start=1):
        t0 = time.perf_counter()
        title, body = load_text(os.path.join(args.dir, fname))
        rec = extract(body)
        ms = (time.perf_counter() - t0) * 1000.0
        rec.update(idx=i, file=fname, title=title, ms=ms)
        records.append(rec)
        print("%-6d %-14s %-16s %-26s %10s %9.1f" % (
            i, fname, rec["date"], rec["company"], rec["amount"], ms))
        for w in rec["warn"]:
            print("      ! %s" % w)
    total = time.perf_counter() - t_all

    n = len(records)
    ok = sum(1 for r in records if r["date"] and r["company"] and r["amount"])
    warns = sum(len(r["warn"]) for r in records)
    print("-" * 92)
    print("样本数 %d 份 | 字段完整 %d 份 | 校验告警 %d 条 | 合计 %.1f 毫秒 | 平均 %.1f 毫秒/份"
          % (n, ok, warns, total * 1000, total * 1000 / n))

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["序号", "文件", "日期", "公司名", "金额(元)", "金额原文", "耗时(ms)", "校验告警"])
            for r in records:
                w.writerow([r["idx"], r["file"], r["date"], r["company"],
                            r["amount"], r["raw"], "%.1f" % r["ms"], "; ".join(r["warn"]) or "无"])
        print("已导出：%s" % args.out)


if __name__ == "__main__":
    main()
