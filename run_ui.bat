@echo off
REM Starts the IDS demo UI. Open http://127.0.0.1:8000 in your browser.
cd /d "%~dp0"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
pause
