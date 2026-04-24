@echo off
cd /d "%~dp0"
rem Force GIL on for Python 3.13+ free-threaded builds (required by mpv/PySide6)
set PYTHON_GIL=1
if exist venv\Scripts\python.exe (
    venv\Scripts\python.exe scripts\cut_inspect.py %*
) else (
    python scripts\cut_inspect.py %*
)
pause
