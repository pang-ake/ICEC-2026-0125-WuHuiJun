# process_logs/ 说明

## 现状（诚实记录）

大纲对项目二的要求是：**连续运行检测日志**（含推理耗时、置信度与误报记录），持续提交到 `process_logs/`。

本目录当前只有两个文件，它们是**真实训练/评估过程的输出指标**，由 `src/train_linear.py` 与 `src/train.py` 直接写出，未经手工修改：

| 文件 | 来源 | 内容 |
|---|---|---|
| `report.json` | `train_linear.py`（三任务分组交叉验证） | 每任务 5 折的张级/组级准确率与标准差、混淆矩阵、多数类基线 |
| `report_amount.json` | `train.py`（端到端微调对照） | amount 任务 3 折结果 + 完整超参 `args`（arch=focalnet... 见文件内 `args` 段） |

示例结构（`report.json` 首条）：

```json
{
  "task": "amount_few_vs_many",
  "n": 175, "n_groups": 18,
  "classes": ["few", "many"], "classes_cn": ["少", "多"],
  "sample_acc": 0.7149, "sample_std": 0.1761,
  "group_acc": 0.8533,  "group_std": 0.1809,
  "folds": [{"sample":0.7353,"group":1.0}, ...],
  "baseline": 0.5086,
  "confusion": [[57,29],[21,68]],
  "clf": "lr"
}
```

## 与大纲要求的差距

| 大纲要求 | 当前状态 |
|---|---|
| 连续多日、逐日提交 | ❌ 目前是一次性训练输出，非逐日累积 |
| 每次推理的耗时与置信度逐条记录 | ❌ 未逐条落盘，只有聚合指标 |
| 误报记录 | ⚠️ 仅有混淆矩阵，无逐帧误报明细 |

**未做美化的原因：** 补一份"看起来很像逐日日志"的文件很容易，但那属于数据造假，违反考核的诚信底线（一票否决项）。所以此目录如实标注缺口。

## 补齐方式（可复现）

在 `src/` 下加一个推理打点脚本，每次检测向 `process_logs/detect_YYYYMMDD.jsonl` 追加一行：

```python
import json, time, datetime, pathlib

def log_one(img_path, conf, pred, elapsed_ms, note=""):
    rec = {
        "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "img": str(img_path),
        "pred": pred,
        "conf": round(float(conf), 4),
        "infer_ms": round(elapsed_ms, 2),
        "note": note,           # 误报/漏报原因，人工补
    }
    p = pathlib.Path("process_logs") / f"detect_{datetime.date.today():%Y%m%d}.jsonl"
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

每天跑一次、`git add process_logs/ && git commit -m "detect log 11-16"`，连续 7 天以上即可满足"连续运行日志"要求。
历史提交时间戳本身就是"逐日"的证据，补记录时不要回填旧日期。

## 复现现有指标

```bash
cd src
python labels.py        # 数据集构成：12 场景 / 204 张 / 21 组
python features.py      # 提特征，CPU 约 50 秒
python train_linear.py  # 三任务交叉验证，约 10 秒 → 重写 report.json
python train.py --task amount --arch mobilenet_v3_small --size 192 --epochs 12 --folds 3
```
