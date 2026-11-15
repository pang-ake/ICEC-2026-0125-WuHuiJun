# -*- coding: utf-8 -*-
"""
动作识别器 —— 按《动作区分规则文档》的四级阈值判定

用法:
    python predict.py "D:/xxx/Raw Data.csv"
    python predict.py "D:/xxx/Raw Data.csv" --win 2.0 --hop 0.5

输出: 整段多数投票结果 + 各窗口判定明细 + 四类票数占比
"""
import sys, argparse
import numpy as np
import pandas as pd
from scipy import signal, stats

FS = 100.0
TH = dict(std_static=0.5, pp_static=3.0, std_fast=5.0, pp_fast=25.0,
          cadence_stair=100.0, crest_stair=2.5, low_stair=0.26)


def lowpass(x, fc, fs=FS):
    b, a = signal.butter(4, fc / (fs / 2), "low")
    return signal.filtfilt(b, a, x)


def features(seg, fs=FS):
    rms = np.sqrt((seg ** 2).mean())
    f, P = signal.welch(seg, fs=fs, nperseg=len(seg))
    band = (f >= 0.5) & (f <= 6.0)
    pk, _ = signal.find_peaks(seg, height=seg.std() * 0.6, distance=int(0.25 * fs))
    return dict(std=seg.std(), pp=seg.max() - seg.min(),
                crest=np.abs(seg).max() / rms if rms > 1e-9 else 0.0,
                low_ratio=P[(f >= 0.5) & (f < 1.2)].sum() / max(P[band].sum(), 1e-12),
                cadence=len(pk) / (len(seg) / fs) * 60)


def classify(ft):
    if ft["std"] < TH["std_static"] and ft["pp"] < TH["pp_static"]:
        return "静止"
    if ft["std"] >= TH["std_fast"] or ft["pp"] >= TH["pp_fast"]:
        return "快走"
    votes = int(ft["cadence"] > TH["cadence_stair"]) \
        + int(ft["crest"] < TH["crest_stair"]) \
        + int(ft["low_ratio"] < TH["low_stair"])
    return "上下楼" if votes >= 2 else "慢走"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Phyphox 导出的 Raw Data.csv 路径")
    ap.add_argument("--win", type=float, default=2.0, help="窗长(秒), 默认 2.0")
    ap.add_argument("--hop", type=float, default=0.5, help="步进(秒), 默认 0.5")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    if df.shape[1] < 4:
        sys.exit("列数不足，需为 Phyphox 加速度(含 g) 的 Raw Data.csv")
    df = df.iloc[:, :4]
    df.columns = ["t", "x", "y", "z"]

    mag = np.sqrt(df.x ** 2 + df.y ** 2 + df.z ** 2).values
    dyn = mag - lowpass(mag, 0.3)

    W, H = int(a.win * FS), int(a.hop * FS)
    rows = []
    for i in range(0, len(dyn) - W + 1, H):
        ft = features(dyn[i:i + W])
        ft["窗起点"] = round(df.t.values[i], 2)
        ft["判定"] = classify(ft)
        rows.append(ft)
    if not rows:
        sys.exit(f"数据太短，至少需要 {a.win} s")

    res = pd.DataFrame(rows)
    print(res.round(2).to_string(index=False))
    print("\n—— 整段判定 ——")
    vc = res["判定"].value_counts()
    for k, v in vc.items():
        print(f"  {k:4s} {v:3d} 窗  {v/len(res)*100:5.1f}%")
    print(f"\n>> 结论: {vc.index[0]}   (最长连续段见上表)")


if __name__ == "__main__":
    main()
