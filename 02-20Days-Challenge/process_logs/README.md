# process_logs/ 说明

## 仓库里有两个 process_logs，职责不同

| 位置 | 内容 | 对应大纲哪一项 |
|---|---|---|
| **本目录** `02-20Days-Challenge/process_logs/` | 建模阶段的评估指标（交叉验证、混淆矩阵） | 第 ③ 项「混淆矩阵与改进报告」 |
| [`../run_log_system/process_logs/`](../run_log_system/process_logs/) | 模型定型后的逐日推理日志（JSONL） | 第 ② 项「连续运行检测日志」 |

两者不可混为一谈：前者回答"模型训得怎么样"，后者回答"系统连续跑得怎么样"。
第 ② 项的实现、字段定义、运行方法和时间线，全部在
[`../run_log_system/README.md`](../run_log_system/README.md)。

---

## 本目录文件

这两个文件是**真实训练/评估过程的输出**，由 `src/train_linear.py` 与 `src/train.py` 直接写出，未经手工修改：

| 文件 | 来源 | 内容 |
|---|---|---|
| `report.json` | `train_linear.py`（三任务分组交叉验证） | 每任务 5 折的张级/组级准确率与标准差、混淆矩阵、多数类基线 |
| `report_amount.json` | `train.py`（端到端微调对照） | amount 任务 3 折结果 + 完整超参 `args` |

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

---

## 连续运行日志在哪

见 [`../run_log_system/process_logs/`](../run_log_system/process_logs/)：

```
run_log_system/process_logs/
├─ 2026-11-15.jsonl   # 每日一个文件，一行一条推理记录
├─ 2026-11-16.jsonl
└─ summary.md         # 由 summarize.py 自动汇总，数字均可从 jsonl 复算
```

**起止时间：2026-11-15 ～ 2026-11-23（封仓前）。** 项目周期是 10-23 ～ 11-24，
日志起点晚于项目起点，原因是前期（10-23 ～ 11-13）做的是数据集采集与模型训练评估，
产出物是模型和指标，本身不存在"运行"这件事。这段时间的过程留痕由 commit 历史、
本目录的 `report.json` 和 `iterations/` 下的三轮实验报告承担。**不描述为"全程连续记录"。**

---

## 关于这一节曾经写过的"缺口说明"

本目录早期版本如实标注过：只有 `report.json` / `report_amount.json`，
**不是**大纲要求的连续逐日推理日志。当时的原话是"补一份看起来很像逐日日志的文件很容易，
但那属于数据造假，违反考核的诚信底线"。

后来的做法是：不编日志，而是把产生日志的系统做出来（`run_log_system/`），
每天真跑一次、真落盘。所以现在 `run_log_system/process_logs/` 里的每一条记录，
都能用 `daily_inference.py` 复跑复现——这也是它可信的原因。

---

## 复现现有指标

```bash
cd src
python labels.py        # 数据集构成：12 场景 / 204 张 / 21 组
python features.py      # 提特征，CPU 约 50 秒
python train_linear.py  # 三任务交叉验证，约 10 秒 → 重写 report.json
python train.py --task amount --arch mobilenet_v3_small --size 192 --epochs 12 --folds 3
```
