@echo off
cd /d "C:\Users\Hp\OneDrive\Desktop\nyaya-rag"
start "" /min cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"
set AGENTIC_CHAT=0
python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
pause
