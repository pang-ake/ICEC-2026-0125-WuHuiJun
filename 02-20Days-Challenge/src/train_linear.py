# -*- coding: utf-8 -*-
"""在预训练特征上做「按连拍组划分」的交叉验证。

三个任务共用同一批特征：
  amount : 少 (few) vs 多 (many)          —— 用户关心的核心任务，175 张 / 18 组
  form   : 散装 / 连体 / 混战              —— 204 张
  light  : 正常光 / 强光 / 阴影 / 弯折      —— 204 张

指标有两个粒度：
  sample-acc 按张统计，同组连拍相互高度相关，偏乐观
  group-acc  按组多数投票，一组=一堆糖，接近真实使用
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import labels as L

HERE = Path(__file__).resolve().parent
FEAT = HERE / "features.npz"
OUT = HERE / "runs"

CN = {
    "amount": {"few": "少", "many": "多", "medium": "中"},
    "form": L.FORM_CN,
    "light": L.LIGHT_CN,
}


def load_feats():
    d = np.load(FEAT, allow_pickle=False)
    return {k: d[k] for k in ("X", "paths", "scenes", "light", "form", "amount", "groups")}


def subset(d, field, keep=None):
    mask = np.ones(len(d["groups"]), bool)
    if keep is not None:
        mask = np.isin(d[field], keep)
    idx = np.where(mask)[0]
    return idx


def group_vote(groups, y_true, y_pred):
    bucket, truth = defaultdict(list), {}
    for g, t, p in zip(groups, y_true, y_pred):
        bucket[g].append(p)
        truth[g] = t
    ok = sum(1 for g, ps in bucket.items() if Counter(ps).most_common(1)[0][0] == truth[g])
    return ok / max(len(bucket), 1), len(bucket)


def evaluate_task(d, field, keep, task_name, clf_kind="lr", folds=5, C=1.0):
    idx = subset(d, field, keep)
    X, y_raw = d["X"][idx], d[field][idx]
    groups = d["groups"][idx]
    classes = sorted(set(y_raw.tolist()))
    cmap = {c: i for i, c in enumerate(classes)}
    y = np.array([cmap[v] for v in y_raw])

    n_groups = len(set(groups.tolist()))
    n_splits = int(min(folds, n_groups))
    if n_splits < 2:
        print("  [跳过] %s：只有 %d 组，无法划分" % (task_name, n_groups))
        return None

    if clf_kind == "svm":
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel="rbf", gamma="scale",
                                                  class_weight="balanced"))
    else:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=4000, class_weight="balanced"))

    gkf = GroupKFold(n_splits=n_splits)
    accs, gaccs, all_t, all_p = [], [], [], []

    for tr, va in gkf.split(X, y, groups):
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[va])
        accs.append(float((p == y[va]).mean()))
        ga, _ = group_vote([groups[i] for i in va], y[va].tolist(), p.tolist())
        gaccs.append(ga)
        all_t += y[va].tolist()
        all_p += p.tolist()

    cm = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(all_t, all_p):
        cm[t, p] += 1

    base = Counter(all_t).most_common(1)[0][1] / len(all_t)
    res = {
        "task": task_name, "n": int(len(y)), "n_groups": n_groups,
        "classes": classes, "classes_cn": [CN[field].get(c, c) for c in classes],
        "sample_acc": float(np.mean(accs)), "sample_std": float(np.std(accs)),
        "group_acc": float(np.mean(gaccs)), "group_std": float(np.std(gaccs)),
        "folds": [{"sample": round(a, 4), "group": round(g, 4)} for a, g in zip(accs, gaccs)],
        "baseline": float(base), "confusion": cm.tolist(), "clf": clf_kind,
    }

    print("\n[%s]  %d 张 / %d 组  类别 %s" % (task_name, len(y), n_groups,
                                          "/".join(res["classes_cn"])))
    print("  张级 %.3f ± %.3f   组级 %.3f ± %.3f   多数类基线 %.3f"
          % (res["sample_acc"], res["sample_std"], res["group_acc"],
             res["group_std"], res["baseline"]))
    print("  各折张级:", "  ".join("%.2f" % a for a in accs))
    print("  各折组级:", "  ".join("%.2f" % g for g in gaccs))
    print("  混淆矩阵（行真实/列预测）:")
    print("  %-8s %s" % ("", " ".join("%6s" % c for c in res["classes_cn"])))
    for i, c in enumerate(res["classes_cn"]):
        print("  %-8s %s" % (c, " ".join("%6d" % v for v in cm[i])))
    return res


def main():
    if not FEAT.exists():
        print("缺少 features.npz，先运行 features.py")
        return
    d = load_feats()
    print("特征 %s，样本 %d，独立组 %d" % (d["X"].shape, len(d["X"]), len(set(d["groups"].tolist()))))

    OUT.mkdir(exist_ok=True)
    results = []

    # 主任务：少 vs 多（排除中量，因为 8~10 颗处在两档之间，口径模糊）
    r = evaluate_task(d, "amount", ["few", "many"], "amount_few_vs_many", "lr", 5)
    if r: results.append(r)

    # 对照任务
    r = evaluate_task(d, "form", None, "form", "lr", 5)
    if r: results.append(r)

    r = evaluate_task(d, "light", None, "light", "lr", 5)
    if r: results.append(r)

    # 消融：去掉「形态」信息还能不能分多少（检验是否靠连体糖作弊）
    loose_only = d["form"] == "loose"
    print("\n" + "=" * 60)
    print("消融：只用纯散糖照片做少/多二分类")
    r2 = evaluate_task(
        {"X": d["X"][loose_only], "amount": d["amount"][loose_only],
         "groups": d["groups"][loose_only], "form": d["form"][loose_only],
         "light": d["light"][loose_only]},
        "amount", ["few", "many"], "amount_loose_only", "lr", 5)
    if r2:
        r2["note"] = "仅纯散糖，排除连体/混战，检验数量本身的可分性"
        results.append(r2)

    (OUT / "report.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n报告已写入 runs/report.json")


if __name__ == "__main__":
    main()
