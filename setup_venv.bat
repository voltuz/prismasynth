@echo off
cd /d "%~dp0"
echo Creating virtual environment...
python -m venv venv
echo Installing dependencies...
venv\Scripts\pip install -r requirements.txt
echo.
echo Setup complete! Run the app with run.bat
pause
