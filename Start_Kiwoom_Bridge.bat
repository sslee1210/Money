@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

title Kiwoom Bridge for Money

set "BRIDGE_DIR=%USERPROFILE%\Desktop\millionaire\bridge"
set "BRIDGE_SCRIPT=kiwoom_bridge_flow.py"
set "BRIDGE_VENV=%BRIDGE_DIR%\.venv32"
set "PYTHON_CMD=py -3-32"

echo.
echo ========================================
echo  Kiwoom Bridge for Money
echo ========================================
echo.
echo [kiwoom] Bridge directory: %BRIDGE_DIR%

if not exist "%BRIDGE_DIR%\%BRIDGE_SCRIPT%" (
    echo [error] Existing Kiwoom bridge script was not found.
    echo [error] Expected: %BRIDGE_DIR%\%BRIDGE_SCRIPT%
    pause
    exit /b 1
)

%PYTHON_CMD% -c "import platform; raise SystemExit(0 if platform.architecture()[0]=='32bit' else 1)" >nul 2>nul
if errorlevel 1 (
    echo [error] 32-bit Python is required for Kiwoom OpenAPI+ ActiveX.
    echo [error] Install 32-bit Python, then run Money_Assistant.bat again.
    pause
    exit /b 1
)

cd /d "%BRIDGE_DIR%"

if not exist "%BRIDGE_VENV%\Scripts\python.exe" (
    echo [setup] Creating 32-bit bridge virtual environment...
    %PYTHON_CMD% -m venv "%BRIDGE_VENV%"
    if errorlevel 1 (
        echo [error] Failed to create 32-bit bridge virtual environment.
        pause
        exit /b 1
    )
)

call "%BRIDGE_VENV%\Scripts\activate.bat"

echo [setup] Checking bridge packages...
python -B -c "import fastapi,uvicorn,PyQt5,win32com.client" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing bridge packages into 32-bit environment...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [error] Failed to install bridge packages.
        pause
        exit /b 1
    )
)

echo [kiwoom] Checking Kiwoom OpenAPI+ ActiveX registration...
python -B -c "import sys; from PyQt5.QtWidgets import QApplication; from PyQt5.QAxContainer import QAxWidget; app=QApplication(sys.argv); ocx=QAxWidget('KHOPENAPI.KHOpenAPICtrl.1'); raise SystemExit(0 if not ocx.isNull() else 1)" >nul 2>nul
if errorlevel 1 (
    echo [error] Kiwoom OpenAPI+ ActiveX could not be created.
    echo [error] Install or repair Kiwoom OpenAPI+ first:
    echo [error] %USERPROFILE%\Downloads\OpenAPISetup.exe
    pause
    exit /b 1
)

set KIWOOM_BRIDGE_PORT=8765
set MAX_REALTIME_CODES=220
set CANDIDATE_REFRESH_MS=90000
set CURRENT_QUOTE_POLL_MS=45000
set CURRENT_QUOTE_BATCH_LIMIT=25
set KIWOOM_EXCHANGE_TYPE=3
set FLOW_WINDOWS_SEC=60,180
set FLOW_AMOUNT_THRESHOLD_MILLION=1000
set FLOW_EVENT_TTL_SEC=900

echo [kiwoom] Starting Kiwoom OpenAPI+ bridge at http://127.0.0.1:8765
echo [kiwoom] If a Kiwoom login window appears, complete login and keep this bridge window open.
python "%BRIDGE_SCRIPT%"

echo.
echo Kiwoom bridge closed.
pause
