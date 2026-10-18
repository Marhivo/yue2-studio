@echo off
chcp 65001 >nul
cd /d "%~dp0"
python -c "import PySide6, numpy, pedalboard, sounddevice" 2>nul || (
    echo Hianyzo csomagok telepitese: PySide6, numpy, pedalboard, sounddevice ...
    python -m pip install PySide6 numpy pedalboard sounddevice
)
python yue2_studio.py
if errorlevel 1 pause
