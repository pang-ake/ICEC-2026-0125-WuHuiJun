# ICEC-2026 选拔考核过程仓库

> 深圳城市职业学院 · 信息与通信学院「智能·创造者」创新实验班（ICEC）
> 2026 级选拔 · GitHub 过程留痕仓库

| 项目 | 内容 |
|---|---|
| 仓库命名 | `ICEC-2026-0125-WuHuiJun` |
| 考生姓名 | Wu HuiJun |
| 学号 | 后四位 0125 |
| GitHub | [pang-ake](https://github.com/pang-ake) |
| 建仓日期 | 2026-11-15 |
| 考核阶段 | 阶段一（基础素养实地测评）+ 阶段二（工坊开放式项目挑战） |

---

## 一、这个仓库是什么

按考核大纲 Step 0 的要求，本仓库用于**持续提交过程数据**，而不是最后一次性上传成品。
所有目录、文件名、`process_logs/` 结构均按大纲规定组织，便于面试时按提交历史追溯。

**证据原则：** 只放真实跑出来的东西——原始 CSV、真实运行输出的 JSON、真实生成的图。
没有跑过的数据不写进来，没有测过的结果不编。

---

## 二、标准目录结构

```
ICEC-2026-0125-WuHuiJun/
├── README.md                    # 本文件：个人简介与选题说明
├── 01-Field-Discovery/          # 阶段一：基础素养实地测评（10 选 2）
│   ├── Task08-Gait-Accelerometer/     # 课题 08 多姿态步态加速度特征采集与对比
│   └── Task10-AI-Batch-Automation/    # 课题 10 日常数字任务耗时审计与 AI 批处理实证
├── 02-20Days-Challenge/         # 阶段二：工坊开放式项目挑战（8 选 1）
│   ├── src/                     # 源码
│   ├── dataset/                 # 自采数据集（204 张 / 12 场景）
│   ├── photos/                  # 采集现场同框照
│   ├── run_log_system/          # 连续运行检测日志（逐日 jsonl + 自动汇总）
│   ├── process_logs/            # 建模阶段的评估指标（混淆矩阵来源）
│   └── iterations/              # 迭代记录、实验报告、配图
└── 03-AI-Collaboration/         # 关键 AI 提问与排障对话记录
```

---

## 三、选题说明

### 阶段一：基础素养实地测评（2/10 选）

| 课题 | 选题理由 | 交付位置 |
|---|---|---|
| **课题 08** 多姿态步态加速度特征采集与对比 | 手上刚好有手机传感器工具，能拿到 100 Hz 真实原始波形；且"四个动作怎么用阈值分开"是可以量化验证的，能给出混淆矩阵而不是只贴几张图 | `01-Field-Discovery/Task08-Gait-Accelerometer/` |
| **课题 10** 日常数字任务耗时审计与 AI 批处理实证 | 校园采购公示的字段抽取是真实高频重复劳动，适合做"人工 vs AI"的可复现对照，并能顺带定出一套人机校验规则 | `01-Field-Discovery/Task10-AI-Batch-Automation/` |

### 阶段二：工坊开放式项目挑战（1/8 选）

| 项目 | 选择 | 对应要求 |
|---|---|---|
| **项目二** 低成本桌面级智能视觉状态检测与抗光照干扰实证装置 | 糖果盘点（少量/多量 + 散装/连体/混战 + 四种光照） | 204 张自采多光照数据集、混淆矩阵、抗光照改进分析 |

选项目二的原因：它的核心难点恰好是**强光、阴影、弯折**这三类真实干扰，而不是把模型堆大。
本仓库的实验结论也集中在"干扰条件下准确率掉在哪里、为什么掉"。

---

## 四、评分维度 → 证据位置对照

| 考核维度 | 分值 | 在本仓库哪里能查到 |
|---|---:|---|
| 阶段一 实地测评 | 20 | `01-Field-Discovery/` 两个课题的 `raw_data/` + 照片 + 报告 |
| 阶段二 项目挑战 | 60 | `02-20Days-Challenge/` 的 `dataset/` `src/` `run_log_system/` `iterations/` |
| 　└ ① ≥200 张多光照数据集（含 EXIF） | 20 | `02-20Days-Challenge/dataset/`（204 张 / 12 场景 / 4 光照） |
| 　└ ② 连续运行检测日志 | 20 | `02-20Days-Challenge/run_log_system/process_logs/`（逐日 JSONL + 汇总） |
| 　└ ③ 混淆矩阵与改进报告 | 20 | `02-20Days-Challenge/iterations/实验报告.md`（含全部混淆矩阵与配图） |
| 阶段三 AI 深度面试 | 20 | `03-AI-Collaboration/` 的对话记录 + 各报告里的"局限与失败分析"章节 |

---

## 五、快速导航

- 步态四动作阈值规则与 90.7% 混淆矩阵 → [`01-Field-Discovery/Task08-Gait-Accelerometer/动作区分规则文档.md`](01-Field-Discovery/Task08-Gait-Accelerometer/动作区分规则文档.md)
- 人工 vs AI 耗时对比（约 1–2 万倍）与校验规则 R1–R9 → [`01-Field-Discovery/Task10-AI-Batch-Automation/REPORT.md`](01-Field-Discovery/Task10-AI-Batch-Automation/REPORT.md)
- 糖果盘点三任务准确率与失败分析 → [`02-20Days-Challenge/iterations/实验报告.md`](02-20Days-Challenge/iterations/实验报告.md)
- 关键 AI 排障对话 → [`03-AI-Collaboration/`](03-AI-Collaboration/)

---

## 六、环境与复现

| 阶段 | 依赖 | 复现命令 |
|---|---|---|
| 课题 08 | Python 3 + numpy / scipy / pandas / matplotlib | `python analysis/analyze.py` |
| 课题 10 | Python 3 + python-docx / openpyxl | `python prompts/AI抽取脚本.py` |
| 项目二 | Python 3.13 + PyTorch 2.12 (CPU) + torchvision + scikit-learn + opencv | 见 `02-20Days-Challenge/src/拍摄清单.md` 同级的实验报告第 6 节 |

项目二为 CPU 可跑，无需 GPU。首次运行会下载 ImageNet 预训练权重（ResNet18 约 45 MB）。

---

## 七、AI 使用与诚信声明

1. 本仓库全部原始数据（CSV / 照片 / docx 样本）均为本人实地采集与拍摄，未经合成或改写。
2. AI 工具用于：代码编写辅助、报告结构梳理、Git 操作排障。所有 AI 参与的排障过程记录在 `03-AI-Collaboration/`。
3. 报告中**保留了全部失败项与局限**（如快走段幅度异常偏大、微信导出导致部分照片丢失 EXIF、组间样本量偏小），未做美化处理。
4. 阶段二数据集为手机原图直接入库，未做压缩或裁剪。

---

*本仓库按考核大纲 Step 0 规范组织，提交历史即过程证据链。*
