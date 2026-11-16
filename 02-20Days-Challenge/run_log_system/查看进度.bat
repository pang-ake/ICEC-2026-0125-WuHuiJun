@echo off
rem ============================================================
rem  Show progress of the daily inference log.
rem  Just double-click this file. Read-only, safe to run anytime.
rem  NOTE: keep CRLF line endings (see .gitattributes).
rem ============================================================

chcp 65001 >nul
cd /d "%~dp0"

set "PY=C:\Users\king2\miniconda3\python.exe"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

if not exist "%PY%" (
    echo ERROR: python not found: %PY%
    pause
    exit /b 1
)

"%PY%" check_progress.py
echo.
pause
