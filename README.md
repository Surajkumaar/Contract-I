# Contract Intelligence

A robust web application for extracting and analyzing structured data from contract PDFs using AI with intelligent fallback mechanisms.

## Project Overview

Contract Intelligence is a full-stack application that helps users extract and analyze key information from contract documents. The application processes PDF contracts and extracts structured data including parties involved, financial details, payment terms, SLAs, and contact information.

### Key Features

- **PDF Contract Processing**: Upload and analyze contract PDFs of any size
- **AI-Powered Extraction**: Uses Mistral AI via OpenRouter API for intelligent data extraction
- **Intelligent Fallback System**: Automatically falls back to regex-based extraction when API is unavailable
- **Multi-layer Error Handling**: Robust system that handles network issues, API failures, and malformed responses
- **Flexible Storage**: Supports both MongoDB and file-based storage with automatic fallback
- **User-Friendly Interface**: Clean React-based UI for easy contract analysis
- **Docker Support**: Fully containerized for easy deployment

## Tech Stack

### Backend
- FastAPI (Python web framework)
- Mistral AI (via OpenRouter API) for contract analysis
- MongoDB (with file-based fallback) for data storage
- PyPDF for PDF text extraction

### Frontend
- React.js
- Material-UI for component styling
- Axios for API communication

## Quick Start (Recommended)

The easiest way to run Contract Intelligence is using Docker:

### Prerequisites
- Docker and Docker Compose
- OpenRouter API key (optional - app works without it using regex fallback)

### Installation Steps

1. **Clone the repository:**
```bash
git clone <repository-url>
cd Contract-I
```

2. **Set up environment variables:**
```bash
# Create .env file in project root (optional but recommended)
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

3. **Start the application:**
```bash
docker-compose up -d --build
```

4. **Access the application:**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- MongoDB: localhost:27017

5. **Stop the application:**
```bash
docker-compose down
```

## Manual Installation (Alternative)

If you prefer to run without Docker:

### Prerequisites
- Python 3.9+
- Node.js 16+ and npm
- MongoDB (optional - app falls back to file storage)

### Backend Setup

1. **Create virtual environment:**
```bash
# Windows
python -m venv cony
cony\Scripts\activate

# Linux/Mac
python -m venv cony
source cony/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cd backend
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

4. **Start backend:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

1. **Install dependencies:**
```bash
cd frontend
npm install
```

2. **Start frontend:**
```bash
npm start
```

The application will be available at http://localhost:3000

## Usage

1. Open your browser and navigate to http://localhost:3000
2. Upload a contract PDF file using the upload button
3. Wait for the analysis to complete
4. View the extracted contract details in a structured format

## Project Structure

```
Contract-I/
├── backend/                # FastAPI backend
│   ├── app/
│   │   ├── main.py         # Main application file
│   │   └── db.py           # Database connection
│   ├── file_storage/       # Fallback storage for contract data
│   └── uploads/            # Directory for uploaded PDF files
├── frontend/               # React frontend
│   ├── public/             # Static files
│   ├── src/                # React source code
│   │   ├── components/     # Reusable UI components
│   │   ├── pages/          # Page components
│   │   └── services/       # API service functions
│   └── package.json        # Frontend dependencies
└── requirements.txt        # Python dependencies
```

## Environment Configuration

### Required Environment Variables

The application uses the following environment variables:

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key for AI analysis | No* | None |
| `MONGO_URL` | MongoDB connection string | No | `mongodb://mongodb:27017` |

*The API key is optional because the application has a robust regex-based fallback system.

### Setting Environment Variables

**For Docker (Recommended):**
```bash
# Create .env file in project root
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

**For Manual Installation:**
```bash
# Create .env file in backend directory
cd backend
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

**For Production:**
Set environment variables directly in your deployment platform or use a secure secrets management system.

## Intelligent Fallback System

Contract Intelligence features a multi-layer fallback system that ensures reliable operation:

### 1. Primary: AI-Powered Extraction
- Uses Mistral AI via OpenRouter API
- Handles complex contract language and context
- Provides the most accurate results

### 2. Secondary: JSON Repair & Recovery
- Attempts to fix malformed API responses
- Handles truncated JSON responses
- Recovers partial data when possible

### 3. Tertiary: Regex-Based Extraction
- Comprehensive pattern matching for contract elements
- Extracts parties, financials, payment terms, SLA, contacts
- Works completely offline without any API dependencies
- Provides structured data in the same format as AI extraction

### Automatic Fallback Triggers
The system automatically falls back when:
- OpenRouter API key is not provided
- Network connectivity issues occur
- API rate limits are exceeded
- JSON parsing fails
- API returns malformed responses

## Docker Architecture

The application consists of three main containers:

### Services
- **Backend**: FastAPI application (Port 8001)
- **Frontend**: React application served by Nginx (Port 3000)
- **MongoDB**: Database for contract storage (Port 27017)

### Volumes
- `mongodb_data`: Persistent MongoDB data
- `./backend/uploads`: PDF file storage
- `./backend/file_storage`: Fallback JSON storage

## Cloud Deployment

The application can also be deployed to cloud services:

- Backend: Hugging Face Spaces (using the provided Dockerfile.hf)
- Frontend: Vercel or any static hosting service

## Troubleshooting

### Common Issues

1. **API Key Issues**
   - **Problem**: "OPENROUTER_API_KEY not set" warning
   - **Solution**: This is normal if you haven't set an API key. The app will use regex fallback extraction.
   - **To fix**: Add your OpenRouter API key to the `.env` file

2. **MongoDB Connection Issues**
   - **Problem**: "MongoDB connection timed out"
   - **Solution**: The app automatically falls back to file-based storage. No action needed.
   - **To fix**: Ensure MongoDB container is running: `docker-compose logs mongodb`

3. **Container Issues**
   - **Problem**: Containers not starting
   - **Solution**: 
     ```bash
     docker-compose down
     docker-compose up -d --build
     ```

4. **PDF Processing Issues**
   - **Problem**: Large PDFs taking too long
   - **Solution**: The app automatically chunks large documents for processing
   - **Fallback**: If AI processing fails, regex extraction will handle the document

5. **Port Conflicts**
   - **Problem**: "Port already in use" errors
   - **Solution**: Stop conflicting services or modify ports in `docker-compose.yml`

### Logs and Debugging

**View application logs:**
```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs frontend
docker-compose logs mongodb
```

**Check container status:**
```bash
docker-compose ps
```

**Restart specific service:**
```bash
docker-compose restart backend
```

## API Endpoints

The backend provides the following REST API endpoints:

- `POST /contracts/upload` - Upload and process a contract PDF
- `GET /contracts/{contract_id}` - Get contract analysis results
- `GET /contracts/{contract_id}/status` - Check processing status
- `GET /health` - Health check endpoint

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (both with and without API key)
5. Submit a pull request

## License

MIT License - See LICENSE file for details
