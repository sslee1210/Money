@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
title Money Assistant

echo.
echo ========================================
echo  Money Assistant
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating virtual environment...
    py -3.11-64 -m venv .venv
    if errorlevel 1 (
        echo.
        echo [error] Python 3.11 64-bit was not found.
        echo Install Python 3.11 64-bit, then run this file again.
        echo.
        pause
        exit /b 1
    )
)

echo [setup] Checking packages...
".venv\Scripts\python.exe" -B -c "import pandas,numpy,requests,FinanceDataReader,pykrx,yfinance,markdown,bs4,lxml,pytest,matplotlib" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing packages. This may take a few minutes on first run...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [error] Package installation failed.
        echo Check your internet connection and Python installation.
        echo.
        pause
        exit /b 1
    )
)

echo.
echo [run] Type a request at the Money prompt.
echo Example: samsung style input is supported in Korean inside the prompt.
echo Example: 005930 Samsung intraday
echo Exit: exit
echo.

".venv\Scripts\python.exe" money_assistant.py

echo.
echo Money Assistant closed.
pause
