# -*- coding: utf-8 -*-
"""
多姿态步态加速度特征采集与对比 —— 分析 + 规则验证
课题 08 | 采样: Phyphox 加速度(含g) 100Hz | 4 类: 静止 / 慢走 / 快走 / 上下楼

输出:
  特征统计表.csv        整段录音级特征
  2秒窗特征分布.csv      滑窗级特征(阈值依据)
  图1~图5 *.png         可视化
  规则验证结果.csv       滑窗级混淆矩阵
"""
import os, glob, warnings
import numpy as np
import pandas as pd
from scipy import signal, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(ROOT)
ACTIONS = ["静止", "慢走", "快走", "上下楼"]
NAME = {"静止": "静止", "慢走": "慢走", "快走": "快走", "上下楼": "上下楼"}
COLOR = {"静止": "#7f8c8d", "慢走": "#2e86de", "快走": "#e74c3c", "上下楼": "#27ae60"}
FS = 100.0
WIN = 200          # 2 秒窗
HOP = 50           # 0.5 秒步进


def load(act):
    f = glob.glob(os.path.join(BASE, act, "*", "Raw Data.csv"))[0]
    df = pd.read_csv(f)
    df.columns = ["t", "x", "y", "z", "abs"]
    return df


def lowpass(x, fc, fs=FS):
    b, a = signal.butter(4, fc / (fs / 2), "low")
    return signal.filtfilt(b, a, x)


def preprocess(df):
    x, y, z = df.x.values, df.y.values, df.z.values
    mag = np.sqrt(x * x + y * y + z * z)      # 合加速度
    grav = lowpass(mag, 0.3)                   # 重力/姿态基线
    dyn = mag - grav                           # 动态分量(去重力)
    return dict(t=df.t.values, x=x, y=y, z=z, mag=mag, grav=grav, dyn=dyn)


def window_features(seg, fs=FS):
    """单个窗口的判别特征"""
    rms = np.sqrt((seg ** 2).mean())
    f, P = signal.welch(seg, fs=fs, nperseg=len(seg))
    band = (f >= 0.5) & (f <= 6.0)
    low = (f >= 0.5) & (f < 1.2)
    walk = (f >= 1.2) & (f <= 3.5)
    pk, _ = signal.find_peaks(seg, height=seg.std() * 0.6, distance=int(0.25 * fs))
    return dict(
        std=seg.std(),
        pp=seg.max() - seg.min(),
        crest=np.abs(seg).max() / rms if rms > 1e-9 else 0.0,
        kurt=stats.kurtosis(seg),
        low_ratio=P[low].sum() / max(P[band].sum(), 1e-12),
        walk_ratio=P[walk].sum() / max(P[band].sum(), 1e-12),
        f_dom=f[band][np.argmax(P[band])],
        cadence=len(pk) / (len(seg) / fs) * 60,
    )


# ============ 阈值规则（核心交付） ============
TH = dict(std_static=0.5, pp_static=3.0,
          std_fast=5.0, pp_fast=25.0,
          cadence_stair=100.0, crest_stair=2.5, low_stair=0.26)


def classify(feat):
    """输入窗口特征, 输出动作标签 + 判据说明"""
    if feat["std"] < TH["std_static"] and feat["pp"] < TH["pp_static"]:
        return "静止"
    if feat["std"] >= TH["std_fast"] or feat["pp"] >= TH["pp_fast"]:
        return "快走"
    votes = int(feat["cadence"] > TH["cadence_stair"]) \
        + int(feat["crest"] < TH["crest_stair"]) \
        + int(feat["low_ratio"] < TH["low_stair"])
    return "上下楼" if votes >= 2 else "慢走"


# ============ 1. 整段特征表 ============
whole, caches = [], {}
for act in ACTIONS:
    p = preprocess(load(act))
    caches[act] = p
    dyn, mag, t = p["dyn"], p["mag"], p["t"]
    dur = t[-1] - t[0]
    f, P = signal.welch(dyn, fs=FS, nperseg=min(512, len(dyn)))
    band = (f >= 0.5) & (f <= 6.0)
    Pb = P[band] / P[band].sum()
    gx, gy, gz = p["x"].mean(), p["y"].mean(), p["z"].mean()
    gn = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    whole.append(dict(
        动作=act, 时长s=round(dur, 2), 样本数=len(t), 采样率Hz=round((len(t) - 1) / dur, 1),
        合加速度均值=round(mag.mean(), 2), 合加速度标准差=round(mag.std(), 2),
        动态std=round(dyn.std(), 2), 动态RMS=round(np.sqrt((dyn ** 2).mean()), 2),
        动态峰峰值=round(dyn.max() - dyn.min(), 2),
        最大单轴std=round(max(p["x"].std(), p["y"].std(), p["z"].std()), 2),
        主频Hz=round(f[band][np.argmax(P[band])], 2),
        步频步每分=round(len(signal.find_peaks(dyn, height=dyn.std() * 0.6,
                                          distance=int(0.25 * FS))[0]) / dur * 60, 1),
        步频带能量占比=round(P[(f >= 1.2) & (f <= 3.5)].sum() / P[band].sum(), 3),
        低频能量占比=round(P[(f >= 0.5) & (f < 1.2)].sum() / P[band].sum(), 3),
        频谱熵=round(-(Pb * np.log2(Pb + 1e-12)).sum() / np.log2(len(Pb)), 3),
        重力模=round(gn, 2), 倾角deg=round(np.degrees(np.arccos(np.clip(gz / gn, -1, 1))), 1),
    ))
whole = pd.DataFrame(whole)
whole.to_csv(os.path.join(ROOT, "特征统计表.csv"), index=False, encoding="utf-8-sig")

# ============ 2. 滑窗特征 + 规则验证 ============
win_rows = []
for act in ACTIONS:
    dyn, t = caches[act]["dyn"], caches[act]["t"]
    for i in range(0, len(dyn) - WIN + 1, HOP):
        r = window_features(dyn[i:i + WIN])
        r.update(动作=act, 窗起点=round(t[i], 2))
        win_rows.append(r)
wf = pd.DataFrame(win_rows)
wf["预测"] = wf.apply(classify, axis=1)
wf.to_csv(os.path.join(ROOT, "2秒窗特征分布.csv"), index=False, encoding="utf-8-sig")

labels = ACTIONS
cm = pd.crosstab(wf["动作"], wf["预测"]).reindex(index=labels, columns=labels).fillna(0).astype(int)
cm.to_csv(os.path.join(ROOT, "规则验证结果.csv"), encoding="utf-8-sig")
acc = (wf["动作"] == wf["预测"]).mean()

# ============ 图1 波形对比 ============
fig, axes = plt.subplots(4, 1, figsize=(12, 10.5))
for ax, act in zip(axes, ACTIONS):
    p = caches[act]
    ax.plot(p["t"], p["mag"], lw=0.6, color=COLOR[act], label="合加速度 |a|")
    ax.plot(p["t"], p["dyn"] + p["mag"].mean(), lw=0.6, color="#f39c12", alpha=0.8,
            label="动态分量 a_dyn")
    ax.axhline(p["mag"].mean(), ls="--", lw=0.8, color="k", alpha=0.35)
    ax.set_title(f"{NAME[act]}    σ(a_dyn)={p['dyn'].std():.2f} m/s²   "
                 f"峰峰值={p['dyn'].max()-p['dyn'].min():.1f} m/s²",
                 fontsize=11, loc="left", color=COLOR[act])
    ax.set_ylabel("m/s²")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, loc="upper right", ncol=2, framealpha=0.85)
axes[-1].set_xlabel("时间 (s)")
fig.suptitle("图 1  四种姿态三轴合加速度波形（Phyphox, 100 Hz, 含重力分量）", fontsize=13)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "图1_四动作波形对比.png"), dpi=170)
plt.close(fig)

# ============ 图2 频谱 ============
fig, ax = plt.subplots(figsize=(11, 4.8))
for act in ACTIONS:
    p = caches[act]
    f, P = signal.welch(p["dyn"], fs=FS, nperseg=min(512, len(p["dyn"])))
    ax.semilogy(f, P, lw=1.3, color=COLOR[act], label=NAME[act])
ax.axvspan(1.2, 3.5, color="#f1c40f", alpha=0.18)
ax.text(1.35, ax.get_ylim()[1] * 0.25, "步行能量带 1.2–3.5 Hz", fontsize=9, color="#b7950b")
ax.set_xlim(0, 8)
ax.set_xlabel("频率 (Hz)")
ax.set_ylabel("功率谱密度 (m/s²)²/Hz")
ax.set_title("图 2  动态分量功率谱对比（Welch 法）")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "图2_功率谱对比.png"), dpi=170)
plt.close(fig)

# ============ 图3 二维可分性 ============
fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
ax = axes[0]
for act in ACTIONS:
    s = wf[wf.动作 == act]
    ax.scatter(s["std"], s["pp"], s=22, alpha=0.55, color=COLOR[act], label=NAME[act],
               edgecolor="none")
ax.axvline(TH["std_static"], ls="--", lw=1, color="#555")
ax.axvline(TH["std_fast"], ls="--", lw=1, color="#555")
ax.axhline(TH["pp_fast"], ls=":", lw=1, color="#555")
ax.text(0.55, 30, "静止阈值\nσ<0.5", fontsize=7.5, color="#555")
ax.text(5.2, 30, "快走阈值\nσ≥5.0", fontsize=7.5, color="#555")
ax.set_xlabel("动态 std (m/s²)"); ax.set_ylabel("峰峰值 (m/s²)")
ax.set_title("(a) 幅度类特征：静止 / 快走 完全可分", fontsize=10.5)
ax.grid(alpha=0.25); ax.legend(fontsize=8)

ax = axes[1]
sub = wf[wf.动作.isin(["慢走", "快走", "上下楼"])]
for act in ["慢走", "快走", "上下楼"]:
    s = sub[sub.动作 == act]
    ax.scatter(s["low_ratio"], s["crest"], s=24, alpha=0.6, color=COLOR[act],
               label=NAME[act], edgecolor="none")
ax.axvline(TH["low_stair"], ls="--", lw=1, color="#555")
ax.axhline(TH["crest_stair"], ls="--", lw=1, color="#555")
ax.plot([0, TH["low_stair"]], [TH["crest_stair"], TH["crest_stair"]], lw=0)
ax.text(0.02, 3.55, "峰均比 < 2.5\n(偏「上下楼」)", fontsize=7.5, color="#555")
ax.text(0.30, 1.9, "低频占比 < 0.26\n(偏「上下楼」)", fontsize=7.5, color="#555")
ax.set_xlabel("低频(<1.2 Hz)能量占比"); ax.set_ylabel("峰均比 crest")
ax.set_title("(b) 波形形状特征：慢走 / 上下楼 重叠区", fontsize=10.5)
ax.grid(alpha=0.25); ax.legend(fontsize=8)
fig.suptitle("图 3  滑窗特征散点（2 s 窗, 0.5 s 步进）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "图3_特征可分性.png"), dpi=170)
plt.close(fig)

# ============ 图4 箱线图 ============
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
for ax, k, lab in zip(axes, ["std", "crest", "low_ratio"],
                      ["动态 std (m/s²)", "峰均比", "低频能量占比 <1.2Hz"]):
    data = [wf[wf.动作 == a][k].values for a in ACTIONS]
    bp = ax.boxplot(data, tick_labels=ACTIONS, patch_artist=True, widths=0.6)
    for patch, a in zip(bp["boxes"], ACTIONS):
        patch.set_facecolor(COLOR[a]); patch.set_alpha(0.55)
    ax.set_title(lab, fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
fig.suptitle("图 4  滑窗级特征分布（阈值上下界依据）", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "图4_滑窗特征箱线图.png"), dpi=170)
plt.close(fig)

# ============ 图5 判定流程 + 混淆矩阵 ============
fig = plt.figure(figsize=(13.5, 5.6))
ax1 = fig.add_axes([0.01, 0.02, 0.58, 0.90]); ax1.axis("off")
ax1.set_xlim(0, 11.2); ax1.set_ylim(0, 10)
def box(x, y, w, h, txt, fc, ec="#333", fs=9.5, bold=False):
    ax1.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.2,
                                zorder=2, joinstyle="round"))
    ax1.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
             zorder=3, weight="bold" if bold else "normal", color="#111")
def arrow(x1_, y1_, x2_, y2_):
    ax1.annotate("", xy=(x2_, y2_), xytext=(x1_, y1_),
                 arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.3), zorder=1)

box(3.2, 9.05, 4.4, 0.8, "输入 2 s 窗（200 点 @100 Hz）", "#ecf0f1", fs=9.5, bold=True)
arrow(5.4, 9.05, 5.4, 8.5)
box(2.7, 7.65, 5.4, 0.85, "|a|=√(x²+y²+z²) → 去 0.3 Hz 低通基线得 a_dyn", "#ecf0f1", fs=8.8)
arrow(5.4, 7.65, 5.4, 7.1)
box(2.4, 6.25, 6.0, 0.85, "① σ(a_dyn) < 0.5 且 峰峰值 < 3.0 ？", "#d5f5e3", fs=9.5)
arrow(8.4, 6.68, 9.35, 6.68); ax1.text(8.55, 6.86, "是", fontsize=9, color="#1e8449")
box(9.3, 6.25, 1.85, 0.85, "静 止", "#27ae60", fs=11, bold=True)
arrow(5.4, 6.25, 5.4, 5.7); ax1.text(5.55, 5.85, "否", fontsize=9, color="#c0392b")
box(2.4, 4.85, 6.0, 0.85, "② σ(a_dyn) ≥ 5.0 或 峰峰值 ≥ 25 ？", "#fdebd0", fs=9.5)
arrow(8.4, 5.28, 9.35, 5.28); ax1.text(8.55, 5.46, "是", fontsize=9, color="#1e8449")
box(9.3, 4.85, 1.85, 0.85, "快 走", "#e74c3c", fs=11, bold=True)
arrow(5.4, 4.85, 5.4, 4.3); ax1.text(5.55, 4.45, "否", fontsize=9, color="#c0392b")
box(1.3, 2.6, 8.2, 1.55,
    "③ 三判据投票（≥2 票判为上下楼）\n"
    "步频 > 100 步/分 ｜ 峰均比 < 2.5 ｜ 低频(<1.2 Hz)能量占比 < 0.26", "#d6eaf8", fs=8.8)
arrow(3.3, 2.6, 3.3, 1.95); ax1.text(3.45, 2.15, "≥ 2 票", fontsize=9, color="#1e8449")
arrow(7.5, 2.6, 7.5, 1.95); ax1.text(7.65, 2.15, "≤ 1 票", fontsize=9, color="#c0392b")
box(1.9, 0.75, 2.8, 1.05, "上下楼", "#27ae60", fs=12, bold=True)
box(6.1, 0.75, 2.8, 1.05, "慢 走", "#2e86de", fs=12, bold=True)
ax1.set_title("图 5  四动作阈值判定流程", fontsize=11.5, loc="left")

ax2 = fig.add_axes([0.63, 0.22, 0.34, 0.62])
im = ax2.imshow(cm.values, cmap="Blues")
ax2.set_xticks(range(4)); ax2.set_yticks(range(4))
ax2.set_xticklabels(labels, fontsize=9); ax2.set_yticklabels(labels, fontsize=9)
ax2.set_xlabel("规则判定结果"); ax2.set_ylabel("真实标签")
for i in range(4):
    for j in range(4):
        ax2.text(j, i, cm.values[i, j], ha="center", va="center", fontsize=11,
                 color="white" if cm.values[i, j] > cm.values.max() * 0.55 else "#111")
ax2.set_title(f"滑窗级混淆矩阵   总准确率 {acc*100:.1f}%", fontsize=11)
fig.savefig(os.path.join(ROOT, "图5_判定流程与混淆矩阵.png"), dpi=170)
plt.close(fig)

# ============ 控制台汇总 ============
pd.set_option("display.width", 240); pd.set_option("display.max_columns", 40)
print("【整段录音特征】"); print(whole.T.to_string())
print("\n【滑窗级规则验证 混淆矩阵】"); print(cm.to_string())
print(f"\n总准确率 = {acc*100:.1f}%  (窗数 {len(wf)})")
print("\n【每类召回率】")
for a in labels:
    s = wf[wf.动作 == a]
    print(f"  {a:4s} {((s.预测==a).mean()*100):5.1f}%   ({int((s.预测==a).sum())}/{len(s)})")
