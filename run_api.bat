@echo off
echo Starting Retention Intelligence API...
echo.
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
