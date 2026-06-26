# Build standalone executable (CPU version)
# Usage: .\packaging\build_cpu.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Aircraft Tracker - CPU Build ===" -ForegroundColor Cyan

# 1. Create clean venv
if (Test-Path build_venv_cpu) { Remove-Item -Recurse -Force build_venv_cpu }
python -m venv build_venv_cpu
.\build_venv_cpu\Scripts\Activate.ps1

# 2. Install CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install pyinstaller

# 3. Build
pyinstaller --clean --onedir `
    --name "AircraftTracker-CPU" `
    --add-data "stabilize;stabilize" `
    --collect-all av `
    --collect-all cv2 `
    --collect-all scipy `
    --hidden-import torch `
    --hidden-import torchvision `
    --hidden-import av `
    --hidden-import cv2 `
    --hidden-import scipy.signal `
    --hidden-import numpy `
    --hidden-import tqdm `
    --exclude-module matplotlib `
    --exclude-module IPython `
    --exclude-module jupyter `
    --exclude-module tkinter.test `
    --console `
    stabilize/main.py

Write-Host "Build complete: dist/AircraftTracker-CPU/" -ForegroundColor Green
