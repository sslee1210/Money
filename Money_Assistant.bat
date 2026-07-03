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
set "KIWOOM_BRIDGE_BAT=%~dp0Start_Kiwoom_Bridge.bat"
set "KIWOOM_FALLBACK_BRIDGE_BAT=%USERPROFILE%\Desktop\millionaire\start-bridge.bat"

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
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --quiet >nul 2>nul
if errorlevel 1 (
    echo [kiwoom] Local bridge is not reachable. Trying to start Kiwoom bridge...
    if not exist "%KIWOOM_BRIDGE_BAT%" (
        set "KIWOOM_BRIDGE_BAT=%KIWOOM_FALLBACK_BRIDGE_BAT%"
    )
    if exist "%KIWOOM_BRIDGE_BAT%" (
        echo [kiwoom] Starting: %KIWOOM_BRIDGE_BAT%
        start "Kiwoom Bridge" "%KIWOOM_BRIDGE_BAT%"
        echo [kiwoom] Waiting for bridge. Complete the Kiwoom login window if it appears.
        for /l %%i in (1,1,90) do (
            ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --quiet >nul 2>nul
            if not errorlevel 1 goto kiwoom_bridge_ready
            timeout /t 2 /nobreak >nul
        )
        echo.
        echo [error] Kiwoom bridge did not become reachable.
        echo [error] Keep the bridge window open, complete Kiwoom login, then run Money_Assistant.bat again.
        pause
        exit /b 1
    ) else (
        echo [kiwoom] Existing bridge launcher not found: %KIWOOM_BRIDGE_BAT%
        echo [error] Kiwoom bridge is required for Money Assistant analysis.
        pause
        exit /b 1
    )
) else (
    goto kiwoom_bridge_ready
)
goto kiwoom_bridge_checked

:kiwoom_bridge_ready
echo [kiwoom] Local bridge is reachable. Checking Kiwoom login status...
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py
for /l %%i in (1,1,180) do (
    ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --require-login --quiet >nul 2>nul
    if not errorlevel 1 goto kiwoom_login_ready
    timeout /t 2 /nobreak >nul
)
echo.
echo [error] Bridge is running, but Kiwoom login is not completed.
echo [error] Complete the Kiwoom login window, then run Money_Assistant.bat again.
pause
exit /b 1

:kiwoom_login_ready
echo [kiwoom] Kiwoom login is ready. Realtime correction will be attempted.
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --require-analysis --quiet >nul 2>nul
if errorlevel 1 (
    echo.
    echo [error] Kiwoom bridge is reachable, but it is not compatible with Money analysis.
    echo [error] Required analysis endpoint is missing. Close the old bridge window and run Money_Assistant.bat again.
    echo [error] Money expects the bridge in: %~dp0kiwoom_bridge_server
    ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py
    pause
    exit /b 1
)

:kiwoom_bridge_checked

echo.
echo [run] Type a request at the Money prompt.
echo Example: Samsung Electronics in Korean is supported.
echo Example: 005930 Samsung Electronics
echo Exit: exit
echo.

".venv\Scripts\python.exe" money_assistant.py

echo.
echo Money Assistant closed.
pause
