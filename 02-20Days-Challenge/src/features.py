# -*- coding: utf-8 -*-
"""用 ImageNet 预训练 ResNet18 提取 204 张图的特征（只前向，不训练）。

204 张 / 21 个独立组的数据量下，全网络微调极易过拟合，
因此这里只做一次前向把每张图压成 512 维向量，再用线性分类器做交叉验证。
附带水平翻转 TTA：原图与镜像图的特征取平均，抵消手持角度带来的抖动。
"""
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

import labels as L

OUT = Path(__file__).resolve().parent / "features.npz"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_tf(size=224):
    return T.Compose([
        T.Resize(int(size * 1.14)),
        T.CenterCrop(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def main():
    recs = L.load()
    print("待提取 %d 张" % len(recs))

    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Identity()          # 输出 512 维全局特征
    model.eval()
    tf = build_tf()

    feats = []
    t0 = time.time()
    with torch.no_grad():
        for i, r in enumerate(recs, 1):
            with Image.open(r["path"]) as im:
                x = tf(im.convert("RGB")).unsqueeze(0)
            f = (model(x) + model(torch.flip(x, dims=[3]))) / 2.0
            feats.append(f.squeeze(0).numpy())
            if i % 50 == 0:
                print("  %d/%d  %.1fs" % (i, len(recs), time.time() - t0))

    X = np.stack(feats).astype(np.float32)
    np.savez_compressed(
        OUT,
        X=X,
        paths=np.array([str(r["path"]) for r in recs]),
        scenes=np.array([r["scene"] for r in recs]),
        light=np.array([r["light"] for r in recs]),
        form=np.array([r["form"] for r in recs]),
        amount=np.array([r["amount"] for r in recs]),
        groups=np.array([r["group"] for r in recs]),
    )
    print("完成 %.1fs  特征矩阵 %s  -> %s" % (time.time() - t0, X.shape, OUT.name))


if __name__ == "__main__":
    main()
