@echo off
title Aircraft Tracker
cd /d "%~dp0"

echo ========================================
echo   Aircraft Tracker - Launching GUI
echo ========================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python 3.12+ from https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo [OK] Python found.

:: Check dependencies
python -c "import cv2,torch,av,scipy,tqdm" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing dependencies...
    pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency install failed. Check network.
        pause
        exit /b 1
    )
)
echo [OK] Dependencies ready.

:: Launch
echo [INFO] Starting GUI...
echo.
python -m stabilize.main --gui

if errorlevel 1 (
    echo.
    echo [ERROR] Application exited with error code %errorlevel%
)
pause
