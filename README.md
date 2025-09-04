# Contract Intelligence

A modern web application for extracting and analyzing structured data from contract PDFs using AI. The system uses Mistral AI via OpenRouter API to extract key information from contracts and presents it in a user-friendly interface.

## Features

- **PDF Contract Upload**: Upload contract PDFs for automated information extraction
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
- MongoDB for data storage
- Multiple fallback strategies for API calls

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

## License

MIT