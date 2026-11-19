#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查今天是否已经完成一轮推理（form + light 共 60 条）。

退出码：
  0 = 今天已完成，调用方应跳过
  1 = 今天还没跑，调用方继续
"""
import datetime
import os
import sys

TARGET = 60  # 30 张 form + 30 张 light

path = os.path.join("process_logs", datetime.date.today().isoformat() + ".jsonl")

n = 0
if os.path.exists(path):
    with open(path, encoding="utf-8") as f:
        n = sum(1 for _ in f)

sys.exit(0 if n >= TARGET else 1)
