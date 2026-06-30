@echo off
chcp 65001 > nul
cd /d %~dp0

echo 조건부 명령형 차트 분석기
echo.
set /p CODE=종목코드 또는 티커 입력:
set /p NAME=종목명 입력(생략 가능):

if "%NAME%"=="" (
  python command_chart_analyzer.py %CODE%
) else (
  python command_chart_analyzer.py %CODE% %NAME%
)

echo.
pause
