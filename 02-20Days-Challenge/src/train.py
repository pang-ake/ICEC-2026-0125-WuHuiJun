# -*- coding: utf-8 -*-
"""糖果数据集训练脚本（CPU 可跑）。

核心约束：划分必须按「连拍组」而不是按「张」。
同一组是同一堆糖在几秒内的连拍帧，若随机划分，验证集里几乎必然出现
与训练集几乎相同的画面，准确率会虚高到 90%+，但换一堆糖就崩。
因此这里用 GroupKFold，并同时报告两个粒度的指标：
  sample-acc : 按张统计（偏乐观，同组内高度相关）
  group-acc  : 按组多数投票（更接近真实使用，一组=一堆糖）

用法：
  python train.py --task form          # 散装/连体/混战 三分类（204 张）
  python train.py --task form --folds 5
  python train.py --task light         # 正常光/强光/阴影/弯折 四分类
  python train.py --task amount        # 少 vs 多（仅正常光散装，样本极少）
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import GroupKFold

import labels as L

OUT = Path(__file__).resolve().parent / "runs"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

TASKS = {
    # task: (标签字段, 类别白名单 or None, 中文名表)
    "form":   ("form",   None, L.FORM_CN),
    "light":  ("light",  None, L.LIGHT_CN),
    "amount": ("amount", None, L.AMOUNT_CN),
}


class CandyDS(Dataset):
    def __init__(self, paths, ys, tf):
        self.paths, self.ys, self.tf = paths, ys, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            img = im.convert("RGB")
        return self.tf(img), self.ys[i]


def build_tf(train, size=224):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(25, fill=(235, 235, 235)),
            T.ColorJitter(0.45, 0.45, 0.35, 0.12),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.6))], p=0.35),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
        ])
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def pick_records(task):
    """按任务筛出可用记录并编码标签。"""
    field, _, _ = TASKS[task]
    recs = L.load()
    if task == "amount":
        # 少 vs 多两类：连体糖按「1 条=1 份」计入少量，混战因散糖多计入多量。
        # 中量（8~10 颗）处在两档之间，口径模糊，直接排除，避免污染边界。
        recs = [r for r in recs if r["amount"] in ("few", "many")]
    uniq = sorted(set(r[field] for r in recs))
    idx = {v: i for i, v in enumerate(uniq)}
    paths = [r["path"] for r in recs]
    ys = [idx[r[field]] for r in recs]
    groups = [r["group"] for r in recs]
    return paths, ys, groups, uniq


def build_model(arch, n_classes):
    """迁移学习模型。mobilenet_v3_small 在 CPU 上比 resnet18 快约 2.5 倍，
    精度对小数据任务几乎无差别，因此作为默认。"""
    if arch == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        m = resnet18(weights=ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        head, stem = m.fc, m.conv1
    else:
        from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
        m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
        head, stem = m.classifier[3], m.features[0]
    for p in stem.parameters():
        p.requires_grad = False
    return m


def run_fold(tr_p, tr_y, va_p, va_y, classes, args, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tl = DataLoader(CandyDS(tr_p, tr_y, build_tf(True, args.size)),
                    batch_size=args.bs, shuffle=True, num_workers=0, drop_last=False)
    vl = DataLoader(CandyDS(va_p, va_y, build_tf(False, args.size)),
                    batch_size=args.bs, shuffle=False, num_workers=0)

    model = build_model(args.arch, len(classes))

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = (-1, None)
    for ep in range(args.epochs):
        model.train()
        for xb, yb in tl:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        sched.step()
        if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
            acc, _ = evaluate(model, vl)
            if acc > best[0]:
                best = (acc, ep + 1)

    model.eval()
    acc, preds = evaluate(model, vl, ret_pred=True)
    return {"acc": acc, "best_ep": best[1], "preds": preds, "val_y": va_y, "model": model}


@torch.no_grad()
def evaluate(model, dl, ret_pred=False):
    model.eval()
    ok = n = 0
    preds, ys = [], []
    for xb, yb in dl:
        p = model(xb).argmax(1)
        ok += (p == yb).sum().item()
        n += yb.numel()
        preds += p.tolist(); ys += yb.tolist()
    acc = ok / max(n, 1)
    return (acc, preds) if ret_pred else (acc, None)


def group_vote(val_p, preds, val_y):
    """把同一连拍组的预测做多数投票，得到「一组=一堆糖」的准确率。"""
    from collections import Counter, defaultdict
    recs = {r["path"]: r for r in L.load()}
    g_pred, g_true = defaultdict(list), {}
    for p, pr, y in zip(val_p, preds, val_y):
        g = recs[p]["group"]
        g_pred[g].append(pr)
        g_true[g] = y
    ok = sum(1 for g, ps in g_pred.items() if Counter(ps).most_common(1)[0][0] == g_true[g])
    return ok / max(len(g_pred), 1), len(g_pred)


def confusion(y_true, y_pred, k):
    m = np.zeros((k, k), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="form", choices=list(TASKS))
    ap.add_argument("--arch", default="mobilenet_v3_small",
                    choices=["mobilenet_v3_small", "resnet18"])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths, ys, groups, classes = pick_records(args.task)
    task_cn = {"form": "形态分类（散装/连体/混战）",
               "light": "光照分类（正常/强光/阴影/弯折）",
               "amount": "数量分类（少 vs 多）"}[args.task]
    print("=" * 72, flush=True)
    print("任务：%s" % task_cn, flush=True)
    print("样本 %d 张，连拍组 %d 个，类别 %s" % (len(paths), len(set(groups)), classes), flush=True)
    from collections import Counter
    print("类别分布（按张）：", dict(Counter(classes[y] for y in ys)), flush=True)
    seen, g_labels = set(), Counter()
    for g, y in zip(groups, ys):
        if g not in seen:
            seen.add(g)
            g_labels[classes[y]] += 1
    print("类别分布（按组）：", dict(g_labels), flush=True)
    print("=" * 72, flush=True)

    n_groups = len(set(groups))
    folds = min(args.folds, n_groups)
    if folds < 2:
        print("组数不足，无法交叉验证", flush=True)
        return

    gkf = GroupKFold(n_splits=folds)
    X = np.arange(len(paths))
    accs, gaccs, all_true, all_pred = [], [], [], []
    t0 = time.time()

    for fi, (tr, va) in enumerate(gkf.split(X, ys, groups), 1):
        tr_p = [paths[i] for i in tr]; tr_y = [ys[i] for i in tr]
        va_p = [paths[i] for i in va]; va_y = [ys[i] for i in va]
        r = run_fold(tr_p, tr_y, va_p, va_y, classes, args, args.seed + fi)
        gacc, ng = group_vote(va_p, r["preds"], va_y)
        accs.append(r["acc"]); gaccs.append(gacc)
        all_true += va_y; all_pred += r["preds"]
        print("fold %d  训练 %3d 张 / 验证 %3d 张（%d 组）  张级 %.3f   组级 %.3f (best_ep %d)"
              % (fi, len(tr), len(va), ng, r["acc"], gacc, r["best_ep"]))

    cm = confusion(all_true, all_pred, len(classes))
    print("-" * 72, flush=True)
    print("张级准确率 %.3f ± %.3f" % (np.mean(accs), np.std(accs)), flush=True)
    print("组级准确率 %.3f ± %.3f   ← 更接近真实使用" % (np.mean(gaccs), np.std(gaccs)), flush=True)
    print("多数类基线 %.3f" % (Counter(all_true).most_common(1)[0][1] / len(all_true)), flush=True)
    print("\n混淆矩阵（行=真实，列=预测）", flush=True)
    print("%-10s %s" % ("", " ".join("%8s" % c[:8] for c in classes)), flush=True)
    for i, c in enumerate(classes):
        print("%-10s %s" % (c[:10], " ".join("%8d" % v for v in cm[i])), flush=True)
    print("\n耗时 %.1f 秒" % (time.time() - t0), flush=True)

    OUT.mkdir(exist_ok=True)
    rep = {"task": args.task, "task_cn": task_cn, "n": len(paths),
           "n_groups": n_groups, "classes": classes,
           "sample_acc": float(np.mean(accs)), "sample_std": float(np.std(accs)),
           "group_acc": float(np.mean(gaccs)), "group_std": float(np.std(gaccs)),
           "folds": [{"sample": a, "group": g} for a, g in zip(accs, gaccs)],
           "confusion": cm.tolist(),
           "args": vars(args)}
    (OUT / ("report_%s.json" % args.task)).write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("报告已写入 runs/report_%s.json" % args.task, flush=True)


if __name__ == "__main__":
    main()
