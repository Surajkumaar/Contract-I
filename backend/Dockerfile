FROM python:3.9-slim

WORKDIR /app

# Copy application files
COPY ./app ./app
COPY ./uploads ./uploads
COPY ./file_storage ./file_storage

# Create necessary directories if they don't exist
RUN mkdir -p uploads file_storage

# Expose the port
EXPOSE 8000

# Command to run the application
CMD ["sh", "-c", "export $(cat /app/.env | grep -v '^#' | xargs -d '\n') && pip install -r /requirements.txt && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
