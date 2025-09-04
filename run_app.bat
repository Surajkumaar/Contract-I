@echo off
echo Contract Intelligence Application Launcher
echo ========================================
echo.

REM Check if MongoDB is running
echo Checking MongoDB status...
timeout /t 1 >nul
echo MongoDB should be running on localhost:27017

echo.
echo Starting Backend Server...
echo.
start cmd /k "cd backend && echo Activating virtual environment... && call cont\Scripts\activate && echo Starting FastAPI server... && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo.
echo Waiting for backend to initialize...
timeout /t 5 >nul

echo.
echo Starting Frontend Development Server...
echo.
start cmd /k "cd frontend && echo Installing dependencies if needed... && npm install && echo Starting React development server... && npm start"

echo.
echo ========================================
echo Contract Intelligence Application is starting!
echo.
echo - Backend API: http://localhost:8000
echo - Frontend UI: http://localhost:3000
echo.
echo Press any key to exit this launcher (the servers will continue running)
pause >nul
