@echo off
rem ============================================================
rem  ICEC 2026 / Project-2 daily inference log
rem  Called by Windows Task Scheduler once a day.
rem  All output is appended to process_logs\run.log
rem
rem  NOTE: this file must keep CRLF line endings. Windows cmd
rem  mis-parses LF-only .bat files (commands get truncated).
rem ============================================================

rem Force UTF-8 so Python's Chinese output is written correctly
chcp 65001 >nul

rem Switch to the script directory (%~dp0 already ends with a backslash)
cd /d "%~dp0"

rem Python interpreter that has torch installed
set "PY=C:\Users\king2\miniconda3\python.exe"

rem Force UTF-8 for Python stdout, otherwise run.log falls back to GBK
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo [%time%] ERROR python not found: %PY% >> "process_logs\run.log"
    exit /b 1
)

if not exist "process_logs" mkdir "process_logs"

echo. >> "process_logs\run.log"
echo ============================================================ >> "process_logs\run.log"
echo [%time%] START daily inference >> "process_logs\run.log"

rem ---- Task 1: form classification (3 classes) ----
"%PY%" daily_inference.py --n 30 --task form >> "process_logs\run.log" 2>&1

rem ---- Task 2: lighting classification (4 classes) ----
"%PY%" daily_inference.py --n 30 --task light >> "process_logs\run.log" 2>&1

rem ---- Regenerate the summary report ----
"%PY%" summarize.py >> "process_logs\run.log" 2>&1

echo [%time%] DONE >> "process_logs\run.log"
exit /b 0
