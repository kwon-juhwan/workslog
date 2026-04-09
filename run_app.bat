@echo off
title 스마트스토어 리뷰 자동화 실행

echo ================================
echo  스마트스토어 리뷰 자동화 실행
echo ================================
echo.

REM 현재 배치 파일이 있는 폴더로 이동
cd /d "%~dp0"

echo [정보] 현재 폴더: %cd%
echo.

REM Python 3.11 고정
set "PY_CMD=C:\Users\아임반\AppData\Local\Programs\Python\Python311\python.exe"

echo [정보] 사용 Python: %PY_CMD%
echo.

REM 필요시 가상환경 활성화 (사용 중이면 주석 해제)
REM call "%cd%\venv\Scripts\activate.bat"

echo [실행] Streamlit 앱 실행: app.py
"%PY_CMD%" -m streamlit run "%cd%\app.py"

echo.
echo -------------------------------
echo 프로그램이 종료되었습니다.
echo 에러 메시지가 있다면 확인 후 알려주세요.
pause