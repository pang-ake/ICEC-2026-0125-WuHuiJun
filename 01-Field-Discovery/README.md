# 阶段一：基础素养实地测评（10 选 2）

> 权重 20 分 ｜ 要求：包含本校实地采样原始数据（`raw_data/`）与本人现场同框照片

## 本目录选题

| 课题 | 核心产出 | 目录 |
|---|---|---|
| **课题 08** 多姿态步态加速度特征采集与对比 | 4 份 100 Hz 原始 CSV、5 张分析图、阈值规则与 90.7% 混淆矩阵、[4 张采集实况截图](Task08-Gait-Accelerometer/photos/) | [`Task08-Gait-Accelerometer/`](Task08-Gait-Accelerometer/) |
| **课题 10** 日常数字任务耗时审计与 AI 批处理实证 | 10 份样本、人工/AI 双份标注表、Prompt 与脚本、人机校验规则 R1–R9、[现场同框照](Task10-AI-Batch-Automation/photos/) | [`Task10-AI-Batch-Automation/`](Task10-AI-Batch-Automation/) |

## 目录规范

每个课题统一为四层，便于面试时直接定位：

```
TaskXX-Name/
├── REPORT.md      # 一页纸结论：做了什么、测到什么、错在哪
├── raw_data/      # 未经修改的原始数据（CSV / docx / xlsx）
├── photos/        # 采集实况截图（Phyphox 界面，证明数据确为实测）
├── figures/       # 由 raw_data 生成的图，可复现
└── analysis/      # 复现脚本 + 中间结果表
```

## 两个课题的共同点

都不是"交一张图说明一个结论"，而是：**先给原始数据 → 再给判定阈值或对照口径 → 最后给错误分析**。
两个课题都在报告里保留了不利于结论的数据（课题 08 的"快走段幅度异常偏大"、课题 10 的"100% 一致率其实是口径被继承"），因为面试追问基本都会打在这两处。
