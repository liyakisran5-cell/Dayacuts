#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
