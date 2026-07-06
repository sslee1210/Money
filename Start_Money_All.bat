@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"
title Money Integrated Launcher

echo.
echo ========================================
echo  Money + Millionaire Integrated Launcher
echo ========================================
echo.

if "%KIWOOM_BRIDGE_URL%"=="" (
    set "KIWOOM_BRIDGE_URL=http://127.0.0.1:8765"
)
set "KIWOOM_BRIDGE_BAT=%~dp0Start_Kiwoom_Bridge.bat"
set "DASHBOARD_DIR=%~dp0dashboard"

echo [kiwoom] Shared bridge URL: %KIWOOM_BRIDGE_URL%
echo [mode] One Kiwoom login, one Money bridge, Money analysis + dashboard share the same data bridge.
if "%MONEY_RESTART_BRIDGE%"=="" (
    set "MONEY_RESTART_BRIDGE=1"
)

if not exist ".venv\Scripts\python.exe" (
    echo [setup] Creating Money virtual environment...
    py -3.11-64 -m venv .venv
    if errorlevel 1 (
        echo [error] Python 3.11 64-bit was not found.
        pause
        exit /b 1
    )
)

echo [setup] Checking Money packages...
".venv\Scripts\python.exe" -B -c "import pandas,numpy,requests,FinanceDataReader,pykrx,yfinance,markdown,bs4,lxml,pytest,matplotlib" >nul 2>nul
if errorlevel 1 (
    echo [setup] Installing Money packages...
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [error] Money package installation failed.
        pause
        exit /b 1
    )
)

if "%MONEY_RESTART_BRIDGE%"=="1" (
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8765" ^| findstr "LISTENING"') do (
        echo [kiwoom] Restarting existing bridge process on port 8765: %%p
        taskkill /PID %%p /F >nul 2>nul
    )
)

echo [kiwoom] Checking shared Money bridge...
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --quiet >nul 2>nul
if errorlevel 1 (
    echo [kiwoom] Shared bridge is not reachable. Starting Money bridge...
    if not exist "%KIWOOM_BRIDGE_BAT%" (
        echo [error] Money bridge launcher not found: %KIWOOM_BRIDGE_BAT%
        pause
        exit /b 1
    )
    start "Money Kiwoom Bridge" "%KIWOOM_BRIDGE_BAT%"
    echo [kiwoom] Waiting for bridge. Complete the Kiwoom login window if it appears.
    for /l %%i in (1,1,90) do (
        ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --quiet >nul 2>nul
        if not errorlevel 1 goto bridge_ready
        timeout /t 2 /nobreak >nul
    )
    echo [error] Money bridge did not become reachable.
    pause
    exit /b 1
)

:bridge_ready
echo [kiwoom] Bridge is reachable. Waiting for Kiwoom login...
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py
for /l %%i in (1,1,180) do (
    ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --require-login --quiet >nul 2>nul
    if not errorlevel 1 goto login_ready
    timeout /t 2 /nobreak >nul
)
echo [error] Bridge is running, but Kiwoom login is not completed.
pause
exit /b 1

:login_ready
echo [kiwoom] Kiwoom login is ready.
".venv\Scripts\python.exe" -B kiwoom_bridge_status.py --require-analysis --quiet >nul 2>nul
if errorlevel 1 (
    echo [error] Shared bridge is reachable, but Money analysis endpoints are missing.
    ".venv\Scripts\python.exe" -B kiwoom_bridge_status.py
    pause
    exit /b 1
)

if exist "%DASHBOARD_DIR%\package.json" (
    if not exist "%DASHBOARD_DIR%\node_modules\vite\bin\vite.js" (
        echo [dashboard] Installing dashboard packages...
        pushd "%DASHBOARD_DIR%"
        call npm install
        if errorlevel 1 (
            popd
            echo [error] Dashboard package installation failed. Install Node.js/npm and try again.
            pause
            exit /b 1
        )
        popd
    )
    for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":5188" ^| findstr "LISTENING"') do (
        echo [dashboard] Stopping existing dashboard process on port 5188: %%p
        taskkill /PID %%p /F >nul 2>nul
    )
    echo [dashboard] Starting dashboard on http://localhost:5188 using the shared Money bridge.
    start "Money Dashboard" cmd /k "cd /d ""%DASHBOARD_DIR%"" && set PORT=5188&& set KIWOOM_BRIDGE_URL=%KIWOOM_BRIDGE_URL%&& set KIWOOM_EXTERNAL_BRIDGE_ONLY=1&& npm run server"
    echo [dashboard] Opening http://localhost:5188 after startup.
    start "" cmd /c "timeout /t 5 /nobreak >nul && start http://localhost:5188/?money_dashboard=1"
) else (
    echo [dashboard] Folder not found: %DASHBOARD_DIR%
    echo [dashboard] Dashboard launch skipped.
)

echo.
echo [run] Money Assistant uses the same bridge. Dashboard: http://localhost:5188
echo.
".venv\Scripts\python.exe" money_assistant.py

echo.
echo Money integrated launcher closed.
pause
