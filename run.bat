@echo off
cd /d "%~dp0"
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe src\main.py %*
) else (
    python src\main.py %*
)
pause
