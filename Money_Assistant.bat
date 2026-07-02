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

if "%KIWOOM_BRIDGE_URL%"=="" (
    set "KIWOOM_BRIDGE_URL=http://127.0.0.1:8765"
)

echo [kiwoom] Bridge URL: %KIWOOM_BRIDGE_URL%

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

echo [kiwoom] Checking local bridge connection...
".venv\Scripts\python.exe" -B -c "import os,socket,urllib.parse; u=urllib.parse.urlparse(os.environ.get('KIWOOM_BRIDGE_URL','')); host=u.hostname or '127.0.0.1'; port=u.port or 80; s=socket.create_connection((host,port),timeout=1.5); s.close()" >nul 2>nul
if errorlevel 1 (
    echo [kiwoom] Local bridge is not reachable. Integrated analysis will keep public-data analysis and limit intraday buy instructions.
    echo [kiwoom] Start your Kiwoom bridge first if you want realtime quote/minute correction.
) else (
    echo [kiwoom] Local bridge is reachable. Realtime correction will be attempted.
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
