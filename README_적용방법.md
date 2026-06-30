# 적용 방법

1. 이 ZIP 파일을 `Money` 저장소 루트에 압축 해제한다.
2. 아래 파일들이 들어가면 된다.
   - `command_chart_analyzer.py`
   - `docs/06_COMMAND_CHART_ANALYZER_RULES.md`
   - `실행_조건부명령형_차트분석.bat`
3. 실행:
   - Git Bash/PowerShell: `python command_chart_analyzer.py 005930 삼성전자`
   - Windows: `실행_조건부명령형_차트분석.bat` 더블클릭
4. 결과:
   - `reports/{종목명}_{코드}/{종목명}_{코드}_조건부명령형_차트분석.md`
   - `reports/{종목명}_{코드}/{종목명}_{코드}_조건부명령형_차트분석.html`

주의: GitHub 커넥터 쓰기 작업이 403으로 차단되어 저장소에 직접 커밋하지 못했다.
