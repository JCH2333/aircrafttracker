# Build standalone executable (CUDA version)
# Usage: .\packaging\build_cuda.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

Write-Host "=== Aircraft Tracker - CUDA Build ===" -ForegroundColor Cyan

# 1. Create clean venv
if (Test-Path build_venv_cuda) { Remove-Item -Recurse -Force build_venv_cuda }
python -m venv build_venv_cuda
.\build_venv_cuda\Scripts\Activate.ps1

# 2. Install CUDA PyTorch + dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
pip install pyinstaller

# 3. Build
pyinstaller --clean --onedir `
    --name "AircraftTracker-CUDA" `
    --add-data "stabilize;stabilize" `
    --collect-all av `
    --collect-all cv2 `
    --collect-all scipy `
    --collect-all torch `
    --collect-all torchvision `
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
    --exclude-module torch.distributed `
    --console `
    stabilize/main.py

Write-Host "Build complete: dist/AircraftTracker-CUDA/" -ForegroundColor Green
