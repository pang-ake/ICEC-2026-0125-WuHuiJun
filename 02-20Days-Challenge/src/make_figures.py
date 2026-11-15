# -*- coding: utf-8 -*-
"""生成实验报告用的全部配图。

输出到 count_tool/figures/：
  fig_samples.png      12 个场景的代表样本（数据集长什么样）
  fig_matrix.png       采集设计矩阵（光照 × 形态）
  fig_confusion.png    三个任务的混淆矩阵
  fig_compare.png      张级/组级准确率 vs 基线
  fig_lum.png          四种光照的实际亮度分布（用数据验证「强光组是否真亮」）
"""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import labels as L

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)
REPORT = json.loads((HERE / "runs" / "report.json").read_text(encoding="utf-8"))

INK = "#1f2933"
GREY = "#7b8794"
ACCENT = "#0f766e"
WARN = "#b45309"


def rd(p, gray=False):
    """支持中文路径的读图。"""
    buf = np.fromfile(str(p), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return im


def readme():
    return L.load()


def fig_samples(recs):
    """12 个场景 × 3 张代表图，检查数据集实际内容。"""
    scenes = list(L.SCENES)
    by = defaultdict(list)
    for r in recs:
        by[r["scene"]].append(r)

    row_h, thumb_w = 1.0, 1.55
    fig = plt.figure(figsize=(11.2, 1.02 * len(scenes) + 1.1), dpi=150)
    fig.patch.set_facecolor("white")

    for i, scene in enumerate(scenes):
        y = len(scenes) - i
        items = by[scene][:3]
        for j, r in enumerate(items):
            im = rd(r["path"])
            if im is None:
                continue
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            ax = fig.add_axes([0.03 + j * (thumb_w + 0.02) / 11.2 * 11.2 / 11.2,
                               (y - 0.88) / (len(scenes) + 1.1),
                               thumb_w / 11.2, 0.82 * row_h / (len(scenes) + 1.1)])
            ax.imshow(im)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#dfe3e8")
            ax.set_aspect("auto")

        light, form, amount, note = L.SCENES[scene]
        gs = len(set(r["group"] for r in by[scene]))
        fig.text(0.545, (y - 0.30) / (len(scenes) + 1.1), scene,
                 fontsize=10.5, color=INK, weight="bold", va="center")
        fig.text(0.545, (y - 0.60) / (len(scenes) + 1.1),
                 "%s / %s / %s　%d 张（%d 组）" % (
                     L.LIGHT_CN[light], L.FORM_CN[form], L.AMOUNT_CN[amount],
                     len(by[scene]), gs),
                 fontsize=8.8, color=GREY, va="center")
        fig.text(0.545, (y - 0.82) / (len(scenes) + 1.1), note,
                 fontsize=8.2, color="#9aa5b1", va="center")

    fig.text(0.03, 1 - 0.42 / (len(scenes) + 1.1), "数据集实地样本（每个场景取前 3 张）",
             fontsize=13, weight="bold", color=INK)
    fig.text(0.03, 1 - 0.66 / (len(scenes) + 1.1),
             "共 204 张 / 12 个场景 / 21 个独立连拍组", fontsize=9, color=GREY)
    fig.savefig(FIG / "fig_samples.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig_samples.png")


def fig_matrix(recs):
    """采集设计矩阵：行=光照，列=形态。"""
    cnt = defaultdict(int)
    for r in recs:
        cnt[(r["light"], r["form"])] += 1
    gcnt = defaultdict(set)
    for r in recs:
        gcnt[(r["light"], r["form"])].add(r["group"])

    lights = ["normal", "strong", "shadow", "bend"]
    forms = ["loose", "linked", "mixed"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=170)
    ax.set_xlim(0, 3); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("white")

    for i, lt in enumerate(lights):
        y = 3 - i
        for j, fm in enumerate(forms):
            n = cnt.get((lt, fm), 0)
            g = len(gcnt.get((lt, fm), ()))
            x0, y0 = j + 0.06, y + 0.10
            if n == 0:
                ax.add_patch(Rectangle((x0, y0), 0.88, 0.80, fc="#fafbfc",
                                       ec="#e4e7eb", lw=0.8, ls="--"))
                ax.text(x0 + 0.44, y0 + 0.40, "未采集", ha="center", va="center",
                        fontsize=9, color="#cbd2d9")
                continue
            fc = "#eef7f4" if g >= 2 else "#fdf3e7"
            ec = ACCENT if g >= 2 else WARN
            ax.add_patch(Rectangle((x0, y0), 0.88, 0.80, fc=fc, ec=ec, lw=1.2))
            ax.text(x0 + 0.44, y0 + 0.50, "%d 张" % n, ha="center", va="center",
                    fontsize=13, weight="bold", color=INK)
            ax.text(x0 + 0.44, y0 + 0.24, "%d 个独立组" % g, ha="center", va="center",
                    fontsize=8.5, color=ec)

    for j, fm in enumerate(forms):
        ax.text(j + 0.5, 4.03, L.FORM_CN[fm], ha="center", va="bottom",
                fontsize=11, weight="bold", color=INK)
    for i, lt in enumerate(lights):
        ax.text(-0.04, 3 - i + 0.5, L.LIGHT_CN[lt], ha="right", va="center",
                fontsize=11, weight="bold", color=INK)

    ax.text(0, -0.30, "橙框 = 该格只有 1 个独立连拍组，相当于 1 个样本；绿框 = 有 2 个及以上",
            fontsize=8.2, color=GREY)
    ax.set_title("采集设计矩阵（光照 × 形态）", fontsize=13, weight="bold",
                 color=INK, pad=18, loc="left", x=-0.16)
    fig.tight_layout()
    fig.savefig(FIG / "fig_matrix.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig_matrix.png")


def fig_confusion():
    tasks = [r for r in REPORT if r["task"] in ("amount_few_vs_many", "form", "light")]
    titles = {"amount_few_vs_many": "任务 A　少 vs 多（175 张 / 18 组）",
              "form": "任务 B　散装 / 连体 / 混战（204 张 / 21 组）",
              "light": "任务 C　四种光照（204 张 / 21 组）"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=170)
    for ax, r in zip(axes, tasks):
        cm = np.array(r["confusion"], float)
        cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r["classes_cn"])))
        ax.set_yticks(range(len(r["classes_cn"])))
        ax.set_xticklabels(r["classes_cn"], fontsize=10)
        ax.set_yticklabels(r["classes_cn"], fontsize=10)
        ax.set_xlabel("预测", fontsize=9, color=GREY)
        ax.set_ylabel("真实", fontsize=9, color=GREY)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cmn[i, j]
                ax.text(j, i, "%d\n%.0f%%" % (cm[i, j], v * 100), ha="center", va="center",
                        fontsize=9, color="white" if v > 0.55 else INK)
        ax.set_title(titles[r["task"]], fontsize=10.5, weight="bold", color=INK, pad=10)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("混淆矩阵（按连拍组交叉验证汇总）", fontsize=13, weight="bold",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG / "fig_confusion.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig_confusion.png")


def fig_compare():
    tasks = [r for r in REPORT if r["task"] in ("amount_few_vs_many", "form", "light")]
    labels_cn = {"amount_few_vs_many": "A 少 vs 多",
                 "form": "B 散装/连体/混战", "light": "C 四种光照"}
    x = np.arange(len(tasks))
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.4, 4.0), dpi=170)
    ax.bar(x - w, [r["sample_acc"] for r in tasks], w, yerr=[r["sample_std"] for r in tasks],
           capsize=4, fc="#b6d7d0", ec=ACCENT, lw=1, label="张级准确率")
    ax.bar(x, [r["group_acc"] for r in tasks], w, yerr=[r["group_std"] for r in tasks],
           capsize=4, fc=ACCENT, ec="#0a5c56", lw=1, label="组级准确率（更接近真实）")
    ax.bar(x + w, [r["baseline"] for r in tasks], w, fc="#e4e7eb", ec="#cbd2d9",
           lw=1, label="多数类基线")
    for i, r in enumerate(tasks):
        ax.text(i, r["group_acc"] + r["group_std"] + 0.03, "%.0f%%" % (r["group_acc"] * 100),
                ha="center", fontsize=9.5, weight="bold", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels([labels_cn[r["task"]] for r in tasks], fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_ylabel("准确率", fontsize=9, color=GREY)
    ax.axhline(0.5, color="#9aa5b1", ls=":", lw=1)
    ax.legend(frameon=False, fontsize=8.8, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), ncol=3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd2d9")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.set_title("三个任务的表现对比（误差棒为 5 折标准差）", fontsize=12,
                 weight="bold", color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_compare.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig_compare.png")


def fig_lum(recs):
    """用亮度分布验证强光组是否真的拍出了强光。"""
    groups = {"normal": [], "strong": [], "shadow": [], "bend": []}
    over = {"normal": [], "strong": [], "shadow": [], "bend": []}
    for r in recs:
        g = rd(r["path"], gray=True)
        if g is None:
            continue
        groups[r["light"]].append(float(g.mean()))
        over[r["light"]].append(float((g > 250).mean() * 100))

    order = ["normal", "strong", "shadow", "bend"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0), dpi=170)

    data = [groups[k] for k in order]
    bp = axes[0].boxplot(data, patch_artist=True, widths=0.55,
                         medianprops=dict(color=INK, lw=1.6),
                         flierprops=dict(marker="o", ms=3, mfc="#cbd2d9", mec="none"))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor("#eef7f4" if order[i] != "strong" else "#fdf3e7")
        patch.set_edgecolor(ACCENT if order[i] != "strong" else WARN)
        patch.set_linewidth(1.2)
    axes[0].set_xticks(range(1, 5))
    axes[0].set_xticklabels([L.LIGHT_CN[k] for k in order], fontsize=10)
    axes[0].set_ylabel("整图平均亮度 (0-255)", fontsize=9, color=GREY)
    axes[0].set_title("各光照组的实际亮度", fontsize=11, weight="bold", color=INK, loc="left")
    for i, k in enumerate(order):
        axes[0].text(i + 1, np.median(groups[k]) + 4, "%.0f" % np.median(groups[k]),
                     ha="center", fontsize=9, color=INK, weight="bold")

    axes[1].bar(range(4), [np.median(over[k]) for k in order], 0.55,
                fc=["#b6d7d0", "#f3c98b", "#b6d7d0", "#b6d7d0"],
                ec=["#0a5c56", WARN, "#0a5c56", "#0a5c56"], lw=1.1)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels([L.LIGHT_CN[k] for k in order], fontsize=10)
    axes[1].set_ylabel("过曝像素占比 (%, 中位数)", fontsize=9, color=GREY)
    axes[1].set_title("各光照组的过曝比例", fontsize=11, weight="bold", color=INK, loc="left")
    axes[1].axhline(1.0, color=WARN, ls="--", lw=1)
    axes[1].text(3.42, 1.15, "1%", fontsize=8, color=WARN, ha="right")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd2d9")
        ax.tick_params(colors=GREY)
        ax.set_facecolor("white")

    fig.suptitle("数据体检：强光组并没有拍出强光", fontsize=13, weight="bold",
                 color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG / "fig_lum.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("fig_lum.png")


if __name__ == "__main__":
    recs = readme()
    fig_matrix(recs)
    fig_samples(recs)
    fig_confusion()
    fig_compare()
    fig_lum(recs)
    print("全部图已生成 ->", FIG)
