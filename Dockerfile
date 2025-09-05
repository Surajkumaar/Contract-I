FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for frontend build
RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    npm install -g npm@latest

# Copy backend requirements and install
COPY backend/requirements.txt /app/backend/
RUN pip install --no-cache-dir -r backend/requirements.txt

# Copy frontend package.json and install dependencies
COPY frontend/package*.json /app/frontend/
RUN cd /app/frontend && npm ci

# Copy the rest of the application
COPY . /app/

# Build the frontend
RUN cd /app/frontend && npm run build

# Create directories for uploads and file storage
RUN mkdir -p /app/backend/uploads /app/backend/file_storage

# Setup nginx
COPY deployment/nginx.conf /etc/nginx/sites-available/default
RUN ln -sf /etc/nginx/sites-available/default /etc/nginx/sites-enabled/default

# Copy startup script
COPY deployment/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Expose port 7860 (Hugging Face Spaces default port)
EXPOSE 7860

# Start the application
CMD ["/app/start.sh"]
