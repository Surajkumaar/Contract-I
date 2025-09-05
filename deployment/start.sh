#!/bin/bash

# Start the backend server in the background
cd /app/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 &

# Start nginx in the foreground
nginx -g "daemon off;"
