@echo off
cd /d "%~dp0"
python -m venv .venv
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend\requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
pause
