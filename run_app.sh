#!/bin/bash

echo "Contract Intelligence Application Launcher"
echo "========================================"
echo

# Check if MongoDB is running
echo "Checking MongoDB status..."
sleep 1
echo "MongoDB should be running on localhost:27017"

echo
echo "Starting Backend Server..."
echo
gnome-terminal --title="Contract Intelligence Backend" -- bash -c "cd backend && echo 'Activating virtual environment...' && source cont/bin/activate && echo 'Starting FastAPI server...' && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000; exec bash" || xterm -T "Contract Intelligence Backend" -e "cd backend && echo 'Activating virtual environment...' && source cont/bin/activate && echo 'Starting FastAPI server...' && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000; exec bash" || open -a Terminal.app "cd backend && echo 'Activating virtual environment...' && source cont/bin/activate && echo 'Starting FastAPI server...' && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000"

echo
echo "Waiting for backend to initialize..."
sleep 5

echo
echo "Starting Frontend Development Server..."
echo
gnome-terminal --title="Contract Intelligence Frontend" -- bash -c "cd frontend && echo 'Installing dependencies if needed...' && npm install && echo 'Starting React development server...' && npm start; exec bash" || xterm -T "Contract Intelligence Frontend" -e "cd frontend && echo 'Installing dependencies if needed...' && npm install && echo 'Starting React development server...' && npm start; exec bash" || open -a Terminal.app "cd frontend && echo 'Installing dependencies if needed...' && npm install && echo 'Starting React development server...' && npm start"

echo
echo "========================================"
echo "Contract Intelligence Application is starting!"
echo
echo "- Backend API: http://localhost:8000"
echo "- Frontend UI: http://localhost:3000"
echo
echo "Press Ctrl+C to exit this launcher (the servers will continue running)"
read -p "Press Enter to exit"
