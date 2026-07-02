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
set "KIWOOM_BRIDGE_BAT=%USERPROFILE%\Desktop\millionaire\start-bridge.bat"

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
    echo [kiwoom] Local bridge is not reachable. Trying to start existing Kiwoom bridge...
    if exist "%KIWOOM_BRIDGE_BAT%" (
        echo [kiwoom] Starting: %KIWOOM_BRIDGE_BAT%
        start "Kiwoom Bridge" "%KIWOOM_BRIDGE_BAT%"
        echo [kiwoom] Waiting for bridge. Complete the Kiwoom login window if it appears.
        for /l %%i in (1,1,30) do (
            ".venv\Scripts\python.exe" -B -c "import os,socket,urllib.parse; u=urllib.parse.urlparse(os.environ.get('KIWOOM_BRIDGE_URL','')); host=u.hostname or '127.0.0.1'; port=u.port or 80; s=socket.create_connection((host,port),timeout=1.0); s.close()" >nul 2>nul
            if not errorlevel 1 goto kiwoom_bridge_ready
            timeout /t 2 /nobreak >nul
        )
        echo [kiwoom] Bridge did not become reachable yet. Integrated analysis will keep public-data analysis and limit intraday buy instructions.
        echo [kiwoom] Keep the bridge window open and finish login, then run analysis again.
    ) else (
        echo [kiwoom] Existing bridge launcher not found: %KIWOOM_BRIDGE_BAT%
        echo [kiwoom] Integrated analysis will keep public-data analysis and limit intraday buy instructions.
    )
) else (
    goto kiwoom_bridge_ready
)
goto kiwoom_bridge_checked

:kiwoom_bridge_ready
echo [kiwoom] Local bridge is reachable. Realtime correction will be attempted.

:kiwoom_bridge_checked

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
