@echo off
title Aircraft Tracker
cd /d "%~dp0"

echo ============================================
echo   Aircraft Tracker v6 - Setup / Launch
echo ============================================
echo.

:: ---- 1. Python ----
echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo   Python not found. Attempting auto-install via winget...
    echo.
    winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements 2>&1
    if errorlevel 1 (
        echo   winget failed or not available.
        echo   Please install Python manually:
        echo     https://www.python.org/downloads/
        pause
        exit /b 1
    )
    echo   Python installed. Please restart this launcher.
    echo.
    pause
    exit /b 0
)
python --version 2>&1
echo.

:: ---- 2. Dependencies ----
echo [2/4] Checking dependencies...
:: Ensure pip is available even if Scripts not on PATH
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo   pip not available. Run: python -m ensurepip
    pause
    exit /b 1
)
python -c "import cv2,torch,av,scipy,tqdm,customtkinter" >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies... (may take a few minutes)
    python -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 (
        echo   [ERROR] Installation failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo   Done.
) else (
    echo   All dependencies found.
)
echo.

:: ---- 3. AI Model ----
echo [3/4] Checking AI model...
python -c "from stabilize.detection.torchvision_detector import TorchvisionDetector; from stabilize.config import StabilizerConfig; d=TorchvisionDetector(StabilizerConfig()); d.warmup(); print('   Model ready.')" 2>&1
if errorlevel 1 (
    echo   [ERROR] Model download failed. Check your internet connection.
    pause
    exit /b 1
)
echo.

:: ---- 4. Launch ----
echo [4/4] Starting GUI...
start "" python -m stabilize.main --gui
exit /b 0
