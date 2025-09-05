# Contract Intelligence

A web application for extracting and analyzing structured data from contract PDFs using Mistral AI via OpenRouter API.

## Project Overview

Contract Intelligence is a full-stack application that helps users extract and analyze key information from contract documents. The application processes PDF contracts and extracts structured data including parties involved, financial details, payment terms, SLAs, and contact information.

### Key Features

- PDF contract upload and processing
- Automatic extraction of structured data using Mistral AI
- User-friendly interface to view and analyze contract details
- Fallback mechanisms for handling large documents and network issues
- Support for both MongoDB and file-based storage

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

## Prerequisites

Before running the application, make sure you have the following installed:

- Python 3.8+ 
- Node.js 14+ and npm
- MongoDB (optional, application will fall back to file-based storage)
- OpenRouter API key (required for AI analysis)

## Installation and Setup

### Clone the Repository

```bash
git clone <repository-url>
cd Contract-I
```

### Backend Setup

1. Create a Python virtual environment:

```bash
# Windows
python -m venv cony
cony\Scripts\activate

# Linux/Mac
python -m venv cony
source cony/bin/activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the backend directory:

```bash
cd backend
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

Replace `your_api_key_here` with your actual OpenRouter API key.

### Frontend Setup

1. Install Node.js dependencies:

```bash
cd frontend
npm install
```

## Running the Application Locally

### Start the Backend Server

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at http://localhost:8000

### Start the Frontend Development Server

```bash
cd frontend
npm install
npm start
```

The frontend application will be available at http://localhost:3000

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

## Deployment

The application can be deployed using:

- Backend: Hugging Face Spaces (using the provided Dockerfile.hf)
- Frontend: Vercel or any static hosting service

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure your OpenRouter API key is correctly set in the `.env` file.

2. **MongoDB Connection**: If MongoDB connection fails, the application will automatically fall back to file-based storage.

3. **PDF Processing Errors**: For very large PDFs, the application uses chunking to process them in parts.

## License

[Specify your license here]
