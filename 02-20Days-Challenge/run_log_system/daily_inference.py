#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""连续运行检测日志 · 每日推理脚本

对指定数量的图片跑一次推理，把每张结果追加写入 process_logs/YYYY-MM-DD.jsonl。

设计要点（对应考核要求）：
  1. 时间戳取真实系统时间 datetime.now().astimezone()，不写死、不按日期推算
  2. 追加写，同一天跑两次就是两批记录，都留在文件里
  3. 正式记录前先跑 3 张 warmup 并丢弃，避免首次 forward 的内核初始化污染耗时
  4. 计时统一用 time.perf_counter()
  5. 中文路径通过 open(rb) + PIL 读取，避开 Windows 下的编码问题
  6. 抽样用「日期做种子 + 分层轮转」，每天换一批，覆盖全部 12 个场景

模型来源与「不虚高」的处理：
  仓库里没有任何 .pth，所以权重不存在时现场训练一次。
  训练按「连拍组」做 StratifiedGroupKFold，为每一折各训练一个线性分类头，
  并把每张图固定分配给「没见过它所在连拍组」的那一折。
  推理时按图片所属组选对应折的头 —— 这就是标准的 out-of-fold 预测，
  等价于「用没见过的模型预测」，因此日志里的准确率与报告中的
  按组交叉验证结果同源（form 87.7% / light 78.3%），不会虚高。
  若直接用全量数据训练再原样推理，准确率会到 97% 以上，与报告矛盾。

用法：
    python daily_inference.py                     # 今天 30 张，form 任务
    python daily_inference.py --n 30 --task light
    python daily_inference.py --task amount --n 20
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torchvision.models import ResNet18_Weights, resnet18

# ---------------------------------------------------------------- 路径与常量

HERE = Path(__file__).resolve().parent          # run_log_system/
ROOT = HERE.parent                              # 项目根（含 dataset/）

# 兼容两种布局：本仓库用 count_tool/，考核仓库用 src/
SRC = ROOT / "count_tool"
if not SRC.exists():
    SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import labels as L  # noqa: E402  复用已有的标签解析，不重新实现

DEFAULT_DATA = ROOT / "dataset"
DEFAULT_OUT = HERE / "process_logs"
DEFAULT_MODELS = HERE / "models"

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
MAX_SPLITS = 5

TASK_CN = {
    "form": L.FORM_CN,      # loose / linked / mixed
    "light": L.LIGHT_CN,    # normal / strong / shadow / bend
    "amount": L.AMOUNT_CN,  # few / medium / many
}

# 抽样轮转以这一天为第 0 天，保证 9 天覆盖全部图片
ROTATION_START = datetime(2026, 11, 15).date()


# ---------------------------------------------------------------- 图像处理

def build_transform():
    return T.Compose([
        T.Resize(int(INPUT_SIZE * 1.14)),
        T.CenterCrop(INPUT_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def load_image(path: Path) -> Image.Image:
    """中文路径安全读取：先以二进制打开再交给 PIL，避开编码问题。"""
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def _pretrained_backbone() -> torch.nn.Module:
    """ImageNet 预训练 ResNet18，去掉 fc，只做 512 维特征提取。"""
    m = resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = torch.nn.Identity()
    m.eval()
    return m


# ---------------------------------------------------------------- 训练与保存

def fold_splits(groups, y, max_splits=MAX_SPLITS) -> int:
    """折数受「最小类的组数」限制，避免某一折训练集里缺类。"""
    g2c = {}
    for g, yy in zip(groups, y):
        g2c[g] = yy
    per_class = Counter(g2c.values())
    return max(2, min(max_splits, min(per_class.values())))


def train_and_save(task: str, recs, classes: list, save_path: Path):
    """提一次特征，按连拍组做分组交叉验证，每折训一个线性头并保存。

    标准化参数折叠进线性层：y = W·((x-μ)/σ) + b  =>  W' = W/σ, b' = b - W·μ/σ
    这样推理时只需一次前向 + 一次线性映射，不需要额外的 scaler 对象。
    """
    print("[train] 未找到权重，开始训练 task=%s 类别=%s" % (task, classes), flush=True)
    t0 = time.perf_counter()

    bb = _pretrained_backbone()
    tf = build_transform()
    idx = {c: i for i, c in enumerate(classes)}

    feats, ys, groups = [], [], []
    with torch.no_grad():
        for r in recs:
            x = tf(load_image(r["path"])).unsqueeze(0)
            feats.append(bb(x).squeeze(0).numpy())
            ys.append(idx[r[task]])
            groups.append(r["group"])

    X = np.stack(feats).astype(np.float32)
    y = np.array(ys)

    n_splits = fold_splits(groups, y)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    heads, group_to_fold = [], {}
    for k, (tr, va) in enumerate(skf.split(X, y, groups)):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y[tr])

        W = clf.coef_.astype(np.float32)
        b = clf.intercept_.astype(np.float32)
        if len(classes) == 2 and W.shape[0] == 1:
            # sklearn 二分类只给一行系数，补成 [logit, 0]，使 softmax 与 sigmoid 等价
            W = np.vstack([W, np.zeros_like(W)])
            b = np.concatenate([b, np.zeros_like(b)])
        W_folded = W / scaler.scale_.astype(np.float32)
        b_folded = b - (W_folded * scaler.mean_.astype(np.float32)).sum(axis=1)

        heads.append({
            "w": torch.from_numpy(W_folded),
            "b": torch.from_numpy(b_folded),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
        })
        for i in va:
            group_to_fold[groups[i]] = k

    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "backbone": bb.state_dict(),
        "heads": heads,
        "group_to_fold": group_to_fold,
        "classes": list(classes),
        "task": task,
        "arch": "resnet18",
        "n_splits": n_splits,
        "trained_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "train_samples": int(len(y)),
    }, save_path)

    print("[train] 完成：%d 张 / %d 组 / %d 折，用时 %.1f 秒 -> %s (%.1f MB)"
          % (len(y), len(set(groups)), n_splits, time.perf_counter() - t0,
             save_path.name, save_path.stat().st_size / 1024 / 1024), flush=True)
    print("[train] 每折训练/验证张数：%s"
          % "，".join("%d/%d" % (h["n_train"], h["n_val"]) for h in heads), flush=True)


def load_model(task: str, model_path: Path, data_dir: Path):
    """加载权重；不存在则训练。返回 (backbone, heads, group_to_fold, classes, 是否新训练)。"""
    if model_path.exists():
        blob = torch.load(model_path, map_location="cpu", weights_only=False)
        bb = resnet18(weights=None)
        bb.fc = torch.nn.Identity()
        bb.load_state_dict(blob["backbone"])
        bb.eval()
        print("[model] 加载已有权重 %s（类别 %s，%d 折，训练于 %s）"
              % (model_path.name, "/".join(blob["classes"]), blob["n_splits"],
                 blob.get("trained_at", "?")), flush=True)
        return bb, blob["heads"], blob["group_to_fold"], blob["classes"], False

    recs = build_records(task, data_dir)
    classes = sorted(set(r[task] for r in recs))
    train_and_save(task, recs, classes, model_path)

    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    bb = resnet18(weights=None)
    bb.fc = torch.nn.Identity()
    bb.load_state_dict(blob["backbone"])
    bb.eval()
    return bb, blob["heads"], blob["group_to_fold"], blob["classes"], True


# ---------------------------------------------------------------- 数据与抽样

def build_records(task: str, data_dir: Path = DEFAULT_DATA):
    """用 labels.py 扫描数据集（目录名即标签），按任务筛选可用样本。"""
    recs = L.load(root=data_dir)
    if task == "amount":
        # 中量（8~10 颗）处在两档之间，口径模糊，排除，避免污染边界
        recs = [r for r in recs if r["amount"] in ("few", "many")]
    return recs


def pick_daily(recs, n: int, seed: str, day_index: int):
    """按「日期种子 + 分层轮转」抽样。

    12 个场景轮流分配名额，保证每天都覆盖全部场景（含表现最差的阴影组）。
    场景内用日期种子打乱后按游标轮转，使相邻几天抽到的图片基本不重复。
    """
    rng = random.Random("seed-%s" % seed)
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(r)

    scenes = sorted(by_scene)                      # 固定顺序，保证可复现
    per = n // len(scenes)
    extra = n - per * len(scenes)

    picked = []
    for i, scene in enumerate(scenes):
        pool = sorted(by_scene[scene], key=lambda r: str(r["path"]))
        rng.shuffle(pool)
        take = per + (1 if i < extra else 0)
        if take <= 0:
            continue
        cursor = (day_index * max(take, 1)) % len(pool)
        for k in range(take):
            picked.append(pool[(cursor + k) % len(pool)])

    rng.shuffle(picked)
    return picked[:n]


# ---------------------------------------------------------------- 推理

def infer_one(bb, heads, g2f, tf, rec, classes, task):
    """对单张图推理。使用「没见过该连拍组」的那一折的头（out-of-fold）。"""
    t0 = time.perf_counter()
    x = tf(load_image(rec["path"])).unsqueeze(0)
    t1 = time.perf_counter()

    with torch.no_grad():
        feat = bb(x)
        fold = g2f.get(rec["group"], 0)
        h = heads[fold]
        logits = F.linear(feat, h["w"], h["b"])
        prob = torch.softmax(logits, dim=1)[0]

    t2 = time.perf_counter()

    probs = {}
    for c, p in zip(classes, prob.tolist()):
        probs[TASK_CN[task][c]] = round(float(p), 4)
    # 四舍五入后总和可能差一点，把误差补到最大项，保证严格等于 1.0
    s = sum(probs.values())
    if abs(s - 1.0) > 1e-9:
        top = max(probs, key=probs.get)
        probs[top] = round(probs[top] + (1.0 - s), 4)

    pred_cn = max(probs, key=probs.get)
    return {
        "pred": pred_cn,
        "confidence": probs[pred_cn],
        "confidence_all": probs,
        "fold": fold,
        "preprocess_ms": round((t1 - t0) * 1000, 2),
        "inference_ms": round((t2 - t1) * 1000, 2),
        "total_ms": round((t2 - t0) * 1000, 2),
    }


def main():
    ap = argparse.ArgumentParser(description="连续运行检测日志 · 每日推理")
    ap.add_argument("--n", type=int, default=30, help="本次跑多少张，默认 30")
    ap.add_argument("--task", default="form", choices=["form", "light", "amount"],
                    help="跑哪个任务，默认 form")
    ap.add_argument("--seed", default=None,
                    help="抽样随机种子，默认取当天日期，保证同一天可复现")
    ap.add_argument("--model", default=None,
                    help="权重路径，默认 models/{task}_resnet18.pth")
    ap.add_argument("--out", default=None, help="输出目录，默认 process_logs/")
    ap.add_argument("--data", default=None, help="数据集目录，默认 ../dataset")
    ap.add_argument("--warmup", type=int, default=3, help="热身张数，结果丢弃")
    args = ap.parse_args()

    now = datetime.now().astimezone()
    today = now.date()
    seed = args.seed or today.isoformat()
    day_index = (today - ROTATION_START).days   # 允许为负，取模后仍是稳定轮转

    out_dir = Path(args.out) if args.out else DEFAULT_OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else DEFAULT_DATA
    if not data_dir.is_dir():
        print("[error] 找不到数据集目录：%s" % data_dir, flush=True)
        sys.exit(2)

    model_path = Path(args.model) if args.model else (DEFAULT_MODELS / ("%s_resnet18.pth" % args.task))

    print("=" * 68, flush=True)
    print("任务 %s　时间 %s　种子 %s（第 %d 天）"
          % (args.task, now.isoformat(timespec="seconds"), seed, day_index), flush=True)

    # 1. 模型
    bb, heads, g2f, classes, freshly = load_model(args.task, model_path, data_dir)

    # 2. 抽样
    recs = build_records(args.task, data_dir)
    if not recs:
        print("[error] 数据集为空或标签不匹配", flush=True)
        sys.exit(2)
    picked = pick_daily(recs, args.n, seed, day_index)
    scenes_hit = sorted(set(r["scene"] for r in picked))
    print("[sample] 抽 %d 张，覆盖 %d 个场景" % (len(picked), len(scenes_hit)), flush=True)

    tf = build_transform()

    # 3. warmup：首次 forward 有内核初始化开销，必须丢弃
    if args.warmup > 0:
        t_w = time.perf_counter()
        with torch.no_grad():
            for r in picked[:args.warmup]:
                bb(tf(load_image(r["path"])).unsqueeze(0))
        print("[warmup] %d 张，用时 %.1f ms（已丢弃）"
              % (min(args.warmup, len(picked)), (time.perf_counter() - t_w) * 1000), flush=True)

    # 4. 正式推理
    rows = []
    for i, r in enumerate(picked, 1):
        res = infer_one(bb, heads, g2f, tf, r, classes, args.task)
        label_cn = TASK_CN[args.task][r[args.task]]
        rows.append({
            # 精确到毫秒：单条推理约 100~150 ms，秒级精度会让多条记录撞在同一秒，
            # 看起来像批量生成，也与「实时逐张检测」的语义不符。
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "task": args.task,
            "image_path": r["path"].relative_to(ROOT).as_posix()
            if ROOT in r["path"].parents else r["path"].as_posix(),
            "label": label_cn,
            "pred": res["pred"],
            "confidence": res["confidence"],
            "confidence_all": res["confidence_all"],
            "preprocess_ms": res["preprocess_ms"],
            "inference_ms": res["inference_ms"],
            "total_ms": res["total_ms"],
            "is_mismatch": res["pred"] != label_cn,
            "scene": r["scene"],          # 便于事后按场景分析
            "group": r["group"],          # 连拍组，避免同组重复计
            "fold": res["fold"],          # 该图由第几折（没见过它的模型）预测
        })
        if i % 10 == 0 or i == len(picked):
            print("  %d/%d" % (i, len(picked)), flush=True)

    # 5. 追加落盘（同一天多批记录都保留）
    log_path = out_dir / ("%s.jsonl" % today.isoformat())
    with open(log_path, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # 6. 终端汇总
    n = len(rows)
    avg_inf = sum(r["inference_ms"] for r in rows) / n
    avg_total = sum(r["total_ms"] for r in rows) / n
    avg_conf = sum(r["confidence"] for r in rows) / n
    miss = sum(1 for r in rows if r["is_mismatch"])
    print("-" * 68, flush=True)
    print("本次 %d 张　平均推理 %.1f ms　平均端到端 %.1f ms　平均置信度 %.3f　误报 %d 张（%.1f%%）"
          % (n, avg_inf, avg_total, avg_conf, miss, miss / n * 100), flush=True)
    print("已追加写入 %s（本次 %d 行，文件共 %d 行）"
          % (log_path.relative_to(ROOT) if ROOT in log_path.parents else log_path,
             n, sum(1 for _ in open(log_path, encoding="utf-8"))), flush=True)
    if freshly:
        print("提示：本次新建了权重文件，之后运行将直接加载。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
