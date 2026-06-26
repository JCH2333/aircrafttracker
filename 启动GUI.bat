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

:: ---- 3. FFmpeg ----
echo [3/5] Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo   FFmpeg not found. Installing via winget...
    winget install Gyan.FFmpeg --accept-source-agreements --accept-package-agreements 2>&1
    if errorlevel 1 (
        echo   winget install failed. Downloading portable FFmpeg...
        powershell -Command "Invoke-WebRequest -Uri 'https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip' -OutFile '%TEMP%\ffmpeg.zip'" 2>&1
        if errorlevel 1 (
            echo   [WARN] FFmpeg download failed. Output will have no audio.
            echo   Install manually: https://ffmpeg.org/download.html
        ) else (
            powershell -Command "Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%~dp0ffmpeg' -Force" 2>&1
            set "PATH=%~dp0ffmpeg\ffmpeg-release-essentials\bin;%PATH%"
            echo   FFmpeg installed to project folder.
        )
    ) else (
        echo   FFmpeg installed via winget.
    )
) else (
    echo   FFmpeg found.
)
echo.

:: ---- 4. AI Model ----
echo [4/5] Checking AI model...
python -c "from stabilize.detection.torchvision_detector import TorchvisionDetector; from stabilize.config import StabilizerConfig; d=TorchvisionDetector(StabilizerConfig()); d.warmup(); print('   Model ready.')" 2>&1
if errorlevel 1 (
    echo   [ERROR] Model download failed. Check your internet connection.
    pause
    exit /b 1
)
echo.

:: ---- 5. Launch ----
echo [5/5] Starting GUI...
start "" python -m stabilize.main --gui
exit /b 0
