@echo off
rem ============================================================
rem  连续运行检测日志 · 每日自动执行
rem  由 Windows 任务计划程序每天 21:00 调用
rem  输出追加到 process_logs\run.log
rem ============================================================

rem 统一 UTF-8，避免中文路径与中文输出乱码
chcp 65001 >nul

rem 切到脚本所在目录（%~dp0 自带结尾反斜杠，路径含空格也能正确处理）
cd /d "%~dp0"

rem Python 解释器：本项目依赖装在 conda 环境里
rem 换机器时只需改这一行，指向装了 torch 的 python.exe
set "PY=C:\Users\king2\miniconda3\python.exe"

if not exist "%PY%" (
    echo [%date% %time%] ERROR 找不到 Python：%PY% >> "process_logs\run.log"
    exit /b 1
)

if not exist "process_logs" mkdir "process_logs"

echo. >> "process_logs\run.log"
echo ============================================================ >> "process_logs\run.log"
echo [%date% %time%] 开始每日推理 >> "process_logs\run.log"

rem ---- 任务一：形态三分类（主任务） ----
"%PY%" daily_inference.py --n 30 --task form >> "process_logs\run.log" 2>&1

rem ---- 任务二：光照四分类（抗光照干扰的证据） ----
"%PY%" daily_inference.py --n 30 --task light >> "process_logs\run.log" 2>&1

rem ---- 重新生成汇总 ----
"%PY%" summarize.py >> "process_logs\run.log" 2>&1

echo [%date% %time%] 完成 >> "process_logs\run.log"
exit /b 0
