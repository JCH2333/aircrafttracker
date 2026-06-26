@echo off
chcp 65001 >nul
title Aircraft Tracker

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未找到 Python，请先安装 Python 3.12+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check dependencies
python -c "import cv2, torch, av, scipy, tqdm" >nul 2>&1
if %errorlevel% neq 0 (
    echo [提示] 缺少依赖，正在安装...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [错误] 依赖安装失败，请检查网络连接
        pause
        exit /b 1
    )
)

:: Launch GUI
cd /d "%~dp0"
python -m stabilize.main --gui

if %errorlevel% neq 0 (
    echo.
    echo [提示] 程序异常退出，请检查上方错误信息
    pause
)
