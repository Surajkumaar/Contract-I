# Contract Intelligence

A modern web application for extracting and analyzing structured data from contract PDFs using AI. The system uses Mistral AI via OpenRouter API to extract key information from contracts and presents it in a user-friendly interface.

## Features

- **PDF Contract Upload**: Upload contract PDFs for automated information extraction
- **Large File Support**: Process PDFs up to 50MB with optimized memory handling
- **Dynamic Field Extraction**: Extract standard contract fields and any additional fields found in the document
- **Real-time Processing Status**: Track extraction progress with status indicators
- **Structured Data View**: View extracted contract data in a well-organized format
- **Responsive UI**: Modern, mobile-friendly user interface

## Architecture

The application consists of two main components:

### Backend (FastAPI)

- RESTful API for contract processing
- Asynchronous background tasks for PDF text extraction
- Integration with Mistral AI via OpenRouter API
- MongoDB for data storage with file-based fallback
- Multiple fallback strategies for API calls
- Optimized large file handling with chunking and memory management

### Frontend (React)

- Modern React application with Material UI
- Contract upload with drag-and-drop support
- Real-time status tracking
- Responsive design for all devices
- Dynamic rendering of additional contract fields

## Setup Instructions

### Prerequisites

- Python 3.8+
- Node.js 14+
- MongoDB
- OpenRouter API key

### Backend Setup

1. Navigate to the backend directory:
   ```
   cd backend
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv cont
   cont\Scripts\activate  # Windows
   source cont/bin/activate  # Linux/Mac
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your OpenRouter API key:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

5. Start the backend server:
   ```
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```
   cd frontend
   ```

2. Install dependencies:
   ```
   npm install
   ```

3. Start the development server:
   ```
   npm start
   ```

4. Access the application at `http://localhost:3000`

## API Endpoints

- `POST /contracts/upload`: Upload a contract PDF
- `GET /contracts/{contract_id}/status`: Get contract processing status
- `GET /contracts/{contract_id}`: Get extracted contract data
- `GET /contracts/{contract_id}/download`: Download the original contract PDF file

## Data Structure

Extracted contract data includes:

- `parties`: Information about parties involved in the contract
- `financials`: Financial details and amounts
- `payment_terms`: Payment schedules and terms
- `sla`: Service Level Agreement details
- `contacts`: Contact information for relevant parties
- `additional_fields`: Any other important information found in the contract

## Development

### Adding New Features

1. Backend: Extend the FastAPI routes in `backend/app/main.py`
2. Frontend: Add new components in the `frontend/src/components` directory

### Testing

Upload sample contracts to test the extraction capabilities. The system is designed to handle various contract formats and extract both standard and non-standard fields.

### Large PDF Handling

The system includes several optimizations for handling large PDFs:

- **Chunked File Upload**: Files are streamed in 1MB chunks to prevent memory issues
- **Memory-Optimized PDF Processing**: Pages are processed individually with garbage collection
- **Document Chunking**: Large documents are split into manageable chunks for LLM processing
- **Result Aggregation**: Results from multiple chunks are intelligently combined
- **Increased Request Limits**: The API can handle files up to 50MB

For best results with very large files (>20MB), allow additional processing time.

## Docker Deployment

### Local Development with Docker Compose

1. Copy the example environment file and set your API key:
   ```
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

2. Build and start the containers:
   ```
   docker-compose up --build
   ```

3. Access the application at `http://localhost:3000`

### Production Deployment

The application includes separate Dockerfiles for the backend and frontend for production deployment:

1. Build the backend image:
   ```
   docker build -t contract-intelligence-backend ./backend
   ```

2. Build the frontend image:
   ```
   docker build -t contract-intelligence-frontend ./frontend
   ```

3. Run the containers with appropriate environment variables and volume mounts.

## Hugging Face Deployment

The application can be deployed to Hugging Face Spaces using the provided configuration:

1. Create a new Space on Hugging Face:
   - Go to https://huggingface.co/spaces
   - Click "New Space"
   - Choose "Docker" as the Space SDK
   - Set the hardware tier (recommend at least CPU-M)

2. Set up environment variables in your Hugging Face Space:
   - `OPENROUTER_API_KEY`: Your OpenRouter API key for LLM access

3. Push your code to the Hugging Face Space repository:
   ```bash
   # Clone your Space repository
   git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME
   
   # Copy your project files
   cp -r /path/to/contract-intelligence/* ./YOUR_SPACE_NAME/
   
   # Rename the Hugging Face Dockerfile
   cd YOUR_SPACE_NAME
   mv Dockerfile.hf Dockerfile
   
   # Commit and push
   git add .
   git commit -m "Initial deployment"
   git push
   ```

4. Hugging Face will automatically build and deploy your application

5. Access your application at `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

## License

MIT