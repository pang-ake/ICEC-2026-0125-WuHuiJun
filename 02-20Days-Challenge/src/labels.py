# -*- coding: utf-8 -*-
"""糖果数据集标签解析 + 连拍分组。

目录名是唯一标签来源（12 个场景目录），从路径解析出三个正交维度：
  light  : normal / strong / shadow / bend     拍摄时的光照条件
  form   : loose / linked / mixed              画面里糖的形态
  amount : few / medium / many                 数量档（仅散装类有意义）

另外从文件名的时间戳（微信导出格式 微信图片_20261114172306.jpg）还原拍摄时刻，
把「时间上连续的几张」聚成一个 group。同一 group 几乎必然是同一堆糖的连拍帧，
划分 train/val 必须整组进、整组出，否则验证集准确率会虚高。
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "dataset"

# 目录 -> (light, form, amount, 该场景的目标颗数说明)
SCENES = {
    "正常光/散装少量":  ("normal", "loose",  "few",    "4~5 颗散糖"),
    "正常光/散装多量":  ("normal", "loose",  "many",   "15~20 颗散糖"),
    "正常光/连体少量":  ("normal", "linked", "few",    "1 条连体糖"),
    "正常光/大混战":    ("normal", "mixed",  "many",   "15 散 + 1 条连体"),
    "强光/散装":        ("strong", "loose",  "medium", "约 10 颗散糖"),
    "强光/连体":        ("strong", "linked", "few",    "1 条连体糖"),
    "强光/大混战":      ("strong", "mixed",  "many",   "10 散 + 1 条连体"),
    "阴影/散装":        ("shadow", "loose",  "medium", "约 8 颗散糖"),
    "阴影/连体":        ("shadow", "linked", "few",    "1 条连体糖"),
    "阴影/混战":        ("shadow", "mixed",  "many",   "8 散 + 1 条连体"),
    "弯折/弯折":        ("bend",   "linked", "few",    "1 条弯折连体"),
    "弯折/弯折和散糖":  ("bend",   "mixed",  "many",   "弯折连体 + 散糖"),
}

IMG_EXT = (".jpg", ".jpeg", ".png")
TS_RE = re.compile(r"(\d{14})")

LIGHT_CN = {"normal": "正常光", "strong": "强光", "shadow": "阴影", "bend": "弯折"}
FORM_CN = {"loose": "纯散糖", "linked": "连体糖", "mixed": "混战"}
AMOUNT_CN = {"few": "少量", "medium": "中量", "many": "多量"}


def parse_ts(name):
    """从文件名取出 14 位时间戳，失败返回 None。"""
    m = TS_RE.search(name)
    return m.group(1) if m else None


def ts_to_sec(ts):
    """20261114172306 -> 秒级整数，仅用于算时间差。"""
    if not ts:
        return None
    d, t = ts[:8], ts[8:]
    return (int(d[6:8]) * 24 + int(t[0:2])) * 3600 + int(t[2:4]) * 60 + int(t[4:6])


def collect(root=ROOT):
    """扫描全部图片，返回记录列表。"""
    recs = []
    for scene, (light, form, amount, note) in SCENES.items():
        d = root / scene
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXT:
                ts = parse_ts(p.name)
                recs.append({
                    "path": p, "scene": scene, "light": light,
                    "form": form, "amount": amount, "note": note,
                    "ts": ts, "sec": ts_to_sec(ts),
                })
    return recs


def make_groups(recs, gap_sec=45):
    """同一场景内，相邻拍摄间隔 <= gap_sec 的连成一组。

    返回 recs 会被就地补上 'group' 字段（字符串 id）。
    阈值 45 秒的选择：同一个姿势/同一堆糖的连拍通常几秒一张，
    换堆、换灯、换角度至少要挪东西，间隔普遍超过 1 分钟。
    """
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)

    for scene, items in by_scene.items():
        items.sort(key=lambda r: r["sec"] if r["sec"] is not None else 0)
        gid, prev = 0, None
        for i, r in enumerate(items):
            if r["sec"] is None:
                r["group"] = "%s#g%02d" % (scene, gid)
                continue
            if prev is None or r["sec"] - prev > gap_sec:
                if prev is not None:
                    gid += 1
            r["group"] = "%s#g%02d" % (scene, gid)
            prev = r["sec"]
    return recs


def load(gap_sec=45, root=ROOT):
    recs = collect(root)
    return make_groups(recs, gap_sec)


def summary(recs):
    from collections import Counter, defaultdict
    print("总数 %d 张\n" % len(recs))
    print("%-18s %-8s %-7s %-6s %5s %5s  %s" % ("场景", "光照", "形态", "数量", "张数", "组数", "颗数口径"))
    print("-" * 84)
    by_scene = defaultdict(list)
    for r in recs:
        by_scene[r["scene"]].append(r)
    for scene in SCENES:
        if scene not in by_scene:
            print("%-18s 目录缺失" % scene)
            continue
        items = by_scene[scene]
        gs = len(set(r["group"] for r in items))
        r0 = items[0]
        print("%-18s %-8s %-7s %-6s %5d %5d  %s" % (
            scene, LIGHT_CN[r0["light"]], FORM_CN[r0["form"]],
            AMOUNT_CN[r0["amount"]], len(items), gs, r0["note"]))

    print("\n维度分布：")
    for key, cn in (("light", LIGHT_CN), ("form", FORM_CN), ("amount", AMOUNT_CN)):
        c = Counter(r[key] for r in recs)
        print("  %-7s %s" % (key, "  ".join("%s=%d" % (cn[k], v) for k, v in c.most_common())))
    print("  组数合计 %d" % len(set(r["group"] for r in recs)))


if __name__ == "__main__":
    summary(load())
