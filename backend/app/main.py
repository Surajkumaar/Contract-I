# app/main.py

import os
import uuid
import asyncio
import json
import pathlib
import ssl
import socket
import certifi
import requests
import urllib3
import re
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel
from typing import Optional
import motor.motor_asyncio
from bson.objectid import ObjectId
from pymongo.errors import ServerSelectionTimeoutError
import httpx
from bson import ObjectId

# Configure proper SSL handling
import ssl
from urllib3.util import create_urllib3_context

# Create a proper SSL context with certificate verification
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = True
ssl_context.verify_mode = ssl.CERT_REQUIRED

# Load system certificates
ssl_context.load_default_certs()

# Load environment variables from .env file if it exists
env_path = pathlib.Path(__file__).parent.parent / '.env'

# Function to read API key directly from .env file
def read_api_key_from_env_file():
    if env_path.exists():
        print(f"Loading environment variables from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                key, value = line.split('=', 1)
                if key == 'OPENROUTER_API_KEY':
                    return value
                os.environ[key] = value
    return None

# ----------------- Configuration -----------------
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "contracts_db"
COLLECTION_NAME = "contracts"

# Try to get API key from environment variable first, then from file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or read_api_key_from_env_file()
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable is not set. API calls will fail.")
MISTRAL_MODEL = "meta-llama/llama-4-maverick:free "  # free model

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- Storage System -----------------
# Define a flag to track if MongoDB is available
USE_MONGO = True

# Create a directory for file-based storage as fallback
FILE_STORAGE_DIR = "file_storage"
os.makedirs(FILE_STORAGE_DIR, exist_ok=True)

# Try to connect to MongoDB
try:
    # Set a short server selection timeout to fail fast if MongoDB is not available
    client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL, serverSelectionTimeoutMS=2000)
    # Don't validate connection immediately - just set up the client
    db = client[DB_NAME]
    contracts_collection = db[COLLECTION_NAME]
    print("✅ MongoDB client initialized - will test connection on first operation")
    
    # We'll validate the connection on first use, not here
    # This avoids blocking startup and allows for MongoDB to become available later
except Exception as e:
    print(f"⚠️ MongoDB client initialization failed: {str(e)}")
    print("⚠️ Falling back to file-based storage")
    USE_MONGO = False
    # Define dummy objects to prevent errors
    client = None
    db = None
    contracts_collection = None

# ----------------- FastAPI App -----------------
app = FastAPI(
    title="Contract Intelligence API",
    # Increase maximum request size to handle large PDFs (100MB)
    max_request_size=104857600
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Models -----------------
class ContractStatus(BaseModel):
    status: str
    progress: int
    score: Optional[int] = None
    error: Optional[str] = None

# ----------------- Helper Functions -----------------
async def resolve_hostname_to_ip(hostname: str) -> str:
    """Try to resolve hostname to IP address using different methods"""
    try:
        # Try standard resolution first
        ip = socket.gethostbyname(hostname)
        return ip
    except socket.gaierror:
        # If that fails, try some known IP addresses for common services
        known_ips = {
            "openrouter.ai": "104.18.6.192",
            "api.openrouter.ai": "104.18.7.192",
            "httpbin.org": "34.205.4.79"
        }
        return known_ips.get(hostname, hostname)

async def test_network_connectivity() -> bool:
    """Test basic network connectivity with multiple fallbacks"""
    test_urls = [
        "https://httpbin.org/get",
        "https://www.google.com",
        "https://www.cloudflare.com"
    ]
    
    # Test with proper SSL verification first
    for url in test_urls:
        try:
            async with httpx.AsyncClient(timeout=10, verify=certifi.where()) as client:
                response = await client.get(url)
                if response.status_code in [200, 301, 302]:
                    return True
        except Exception:
            continue
    
    # If SSL verification fails, try with system SSL context
    for url in test_urls:
        try:
            async with httpx.AsyncClient(timeout=10, verify=ssl_context) as client:
                response = await client.get(url)
                if response.status_code in [200, 301, 302]:
                    return True
        except Exception:
            continue
    
    return False

async def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF with memory optimization"""
    reader = PdfReader(file_path)
    text = ""
    # Process one page at a time to reduce memory usage
    for page in reader.pages:
        text += page.extract_text() or ""
        # Free up memory by clearing page cache
        page._objects = None
    return text

def chunk_text(text: str, chunk_size: int = 8000, overlap: int = 500) -> list[str]:
    """Split text into chunks suitable for LLM processing"""
    if not text or len(text) <= chunk_size:
        return [text] if text else [""]
        
    chunks = []
    start = 0
    while start < len(text):
        # Take a chunk of text
        end = min(start + chunk_size, len(text))
        
        # Try to find a good breaking point (newline or period)
        if end < len(text):
            # Look for a newline or period to break at
            for break_char in ['\n\n', '\n', '. ', ', ', ' ']:
                break_pos = text.rfind(break_char, start, end)
                if break_pos > start + chunk_size // 2:  # Ensure chunk isn't too small
                    end = break_pos + len(break_char)
                    break
        
        chunks.append(text[start:end])
        # Start next chunk with overlap for context
        start = max(start, end - overlap)
    
    return chunks

async def query_mistral_llm(contract_text: str) -> dict:
    """
    Query the Mistral LLM via OpenRouter API with support for large documents
    Falls back to regex extraction if API is unavailable
    """
    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY not set, using regex fallback extraction")
        return extract_contract_data_with_regex(contract_text)
    
    # Check if contract_text is None or empty
    if not contract_text:
        print("⚠️ Contract text is empty or None, returning empty extraction")
        return extract_contract_data_with_regex("")
    
    try:
        # Check if text needs to be chunked (over 15K characters is likely to exceed token limits)
        if len(contract_text) > 15000:
            print(f"Contract text is large ({len(contract_text)} chars), chunking for processing")
            return await process_large_document(contract_text)
        else:
            # Process normally for smaller documents
            return await query_mistral_llm_single(contract_text)
    except Exception as e:
        print(f"🔄 OpenRouter API failed: {str(e)}")
        print("🔄 Falling back to regex extraction...")
        return extract_contract_data_with_regex(contract_text)

async def process_large_document(contract_text: str) -> dict:
    """Process a large document by chunking and combining results"""
    try:
        # Split text into manageable chunks
        chunks = chunk_text(contract_text)
        print(f"Split document into {len(chunks)} chunks for processing")
        
        # Process each chunk to extract information
        chunk_results = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}...")
            try:
                # Process each chunk with a simplified prompt focused on extraction
                chunk_result = await query_mistral_llm_single(
                    chunk, 
                    is_chunk=True, 
                    chunk_num=i+1, 
                    total_chunks=len(chunks)
                )
                chunk_results.append(chunk_result)
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
        
        # If no chunks were successfully processed, fall back to regex
        if not chunk_results:
            print("🔄 All chunk processing failed, using regex fallback for full document")
            return extract_contract_data_with_regex(contract_text)
        
        # Combine results from all chunks
        combined_result = combine_chunk_results(chunk_results)
        return combined_result
    except Exception as e:
        print(f"🔄 Large document processing failed: {str(e)}")
        print("🔄 Falling back to regex extraction for full document...")
        return extract_contract_data_with_regex(contract_text)

def clean_extracted_text(text: str) -> str:
    """Clean and normalize extracted text"""
    if not text:
        return ""  # Return empty string instead of None
    # Remove excessive whitespace, newlines, and normalize
    cleaned = re.sub(r'\s+', ' ', text.strip())
    # Remove common PDF artifacts
    cleaned = re.sub(r'[\n\r\t]+', ' ', cleaned)
    return cleaned.strip()

def is_valid_party_name(name: str) -> bool:
    """Validate if extracted text is a legitimate party name"""
    if not name or len(name) < 3:
        return False
    # Reject if mostly numbers or single words that are too generic
    if re.match(r'^\d+$', name) or name.lower() in ['may', 'shall', 'will', 'the', 'and', 'or', 'of', 'to', 'in', 'for', 'with']:
        return False
    # Reject if contains too many special characters
    if len(re.findall(r'[^a-zA-Z0-9\s&.,\-]', name)) > len(name) * 0.3:
        return False
    return True

def extract_contract_data_with_regex(contract_text: str) -> dict:
    """
    Fallback function to extract contract data using regex patterns
    when OpenRouter API is not available or fails
    Follows the same structure and format as LLM extraction
    """
    print("🔍 Using regex fallback extraction...")
    
    # Initialize result structure matching LLM output format exactly
    extracted_data = {
        "parties": [],
        "financials": {},
        "payment_terms": {},
        "sla": {},
        "contacts": [],
        "additional_fields": {}
    }
    
    # Clean the contract text first
    contract_text = re.sub(r'\n\s*\n', '\n', contract_text)  # Remove excessive line breaks
    contract_text = re.sub(r'\s{3,}', ' ', contract_text)    # Normalize multiple spaces
    
    # Extract parties/companies with improved patterns
    party_patterns = [
        # Company names with legal suffixes
        r'([A-Z][a-zA-Z0-9\s&.,\-]{2,50})\s+(?:Inc|LLC|Corp|Corporation|Limited|Ltd|Company)\b',
        # Between X and Y pattern
        r'between\s+([A-Z][a-zA-Z0-9\s&.,\-]{2,50})\s+(?:and|&)\s+([A-Z][a-zA-Z0-9\s&.,\-]{2,50})',
        # Client/Contractor with colon
        r'(?:Client|Customer|Contractor|Vendor|Supplier)\s*:\s*([A-Z][a-zA-Z0-9\s&.,\-]{3,50})',
        # Company names in quotes
        r'"([A-Z][a-zA-Z0-9\s&.,\-]{3,50})"',
        # Proper nouns that look like company names (Title Case)
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}(?:\s+(?:Inc|LLC|Corp|Corporation|Limited|Ltd|Company))?)\b'
    ]
    
    parties_found = set()
    for pattern in party_patterns:
        matches = re.findall(pattern, contract_text, re.MULTILINE)
        for match in matches:
            if isinstance(match, tuple):
                for party in match:
                    party = clean_extracted_text(party)
                    if is_valid_party_name(party):
                        parties_found.add(party)
            else:
                party = clean_extracted_text(match)
                if is_valid_party_name(party):
                    parties_found.add(party)
    
    # Format parties as structured objects like LLM output
    formatted_parties = []
    for party in list(parties_found)[:5]:  # Limit to 5 most relevant parties
        # Determine role based on context around the party name
        role = "party"
        party_context = ""
        
        # Find context around party name
        party_matches = list(re.finditer(re.escape(party), contract_text, re.IGNORECASE))
        if party_matches:
            start = max(0, party_matches[0].start() - 100)
            end = min(len(contract_text), party_matches[0].end() + 100)
            party_context = contract_text[start:end].lower()
        
        if re.search(r'client|customer|buyer|purchaser', party_context):
            role = "client"
        elif re.search(r'contractor|vendor|supplier|seller|provider', party_context):
            role = "contractor"
        elif re.search(r'company|corporation|business|entity', party_context):
            role = "company"
        
        formatted_parties.append({
            "name": party,
            "role": role,
            "address": None
        })
    
    extracted_data["parties"] = formatted_parties
    
    # Extract financial information
    financial_patterns = [
        r'\$([0-9,]+(?:\.[0-9]{2})?)',
        r'([0-9,]+(?:\.[0-9]{2})?)\s*dollars?',
        r'amount\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'total\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'price\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'fee\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'cost\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'value\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'sum\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'payment\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'compensation\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'salary\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'budget\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'invoice\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'bill\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'charge\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'rate\s*[:\-]?\s*\$?([0-9,]+(?:\.[0-9]{2})?)',
        r'([0-9,]+(?:\.[0-9]{2})?)\s*per\s+(?:hour|day|week|month|year)',
        r'(?:usd|eur|gbp|cad|aud)\s*([0-9,]+(?:\.[0-9]{2})?)',
        r'([0-9,]+(?:\.[0-9]{2})?)\s*(?:usd|eur|gbp|cad|aud)'
    ]
    
    amounts_found = []
    for pattern in financial_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        for match in matches:
            amount = match.replace(',', '')
            try:
                float_amount = float(amount)
                if 0 < float_amount < 1000000000:  # Reasonable range
                    amounts_found.append(amount)
            except ValueError:
                continue
    
    # Format financials to match LLM structure
    if amounts_found:
        # Find the largest amount as likely total value
        total_value = max(amounts_found, key=lambda x: float(x.replace(',', ''))) if amounts_found else None
        
        # Detect currency from text
        currency = "USD"  # Default
        currency_patterns = [
            (r'\$', 'USD'),
            (r'€|EUR|euro', 'EUR'),
            (r'£|GBP|pound', 'GBP'),
            (r'CAD|canadian', 'CAD'),
            (r'AUD|australian', 'AUD')
        ]
        for pattern, curr in currency_patterns:
            if re.search(pattern, contract_text, re.IGNORECASE):
                currency = curr
                break
        
        extracted_data["financials"] = {
            "amounts": amounts_found[:10],
            "currencies": [currency],
            "total_value": total_value,
            "currency": currency
        }
    else:
        extracted_data["financials"] = None
    
    # Extract payment terms
    payment_patterns = [
        r'payment\s+(?:due|terms?)\s*[:\-]?\s*([^\n\.]{1,100})',
        r'due\s+(?:date|on)\s*[:\-]?\s*([^\n\.]{1,50})',
        r'net\s+(\d+)\s*days?',
        r'within\s+(\d+)\s*(?:days?|weeks?|months?)',
        r'(?:monthly|quarterly|annually|yearly)\s+payment',
        r'installment\s*[:\-]?\s*([^\n\.]{1,100})',
        r'milestone\s*[:\-]?\s*([^\n\.]{1,100})'
    ]
    
    payment_terms_found = []
    for pattern in payment_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        payment_terms_found.extend(matches)
    
    # Format payment terms to match LLM structure with better extraction
    if payment_terms_found:
        payment_schedule = None
        payment_methods = []
        penalties = None
        
        for term in payment_terms_found:
            term = clean_extracted_text(term)
            
            # Skip if term is too short
            if len(term) < 5:
                continue
                
            if re.search(r'net\s+\d+|within\s+\d+\s*(?:days?|weeks?)', term, re.IGNORECASE):
                payment_schedule = term
            elif re.search(r'monthly|quarterly|annually|installment|milestone', term, re.IGNORECASE):
                if not payment_schedule:  # Don't overwrite net terms
                    payment_schedule = term
            elif re.search(r'penalty|late\s+fee|interest|overdue', term, re.IGNORECASE):
                penalties = term
            elif re.search(r'wire\s+transfer|check|credit\s+card|bank\s+transfer|ach', term, re.IGNORECASE):
                payment_methods.append(term)
        
        extracted_data["payment_terms"] = {
            "schedule": payment_schedule,
            "methods": payment_methods if payment_methods else None,
            "penalties": penalties
        }
    else:
        extracted_data["payment_terms"] = None
    
    # Extract SLA/performance terms
    sla_patterns = [
        r'service\s+level\s*[:\-]?\s*([^\n\.]{1,100})',
        r'performance\s*[:\-]?\s*([^\n\.]{1,100})',
        r'delivery\s+(?:time|date)\s*[:\-]?\s*([^\n\.]{1,50})',
        r'completion\s+(?:time|date)\s*[:\-]?\s*([^\n\.]{1,50})',
        r'deadline\s*[:\-]?\s*([^\n\.]{1,50})',
        r'turnaround\s*[:\-]?\s*([^\n\.]{1,50})',
        r'response\s+time\s*[:\-]?\s*([^\n\.]{1,50})',
        r'availability\s*[:\-]?\s*([^\n\.]{1,50})',
        r'uptime\s*[:\-]?\s*([^\n\.]{1,50})'
    ]
    
    sla_terms_found = []
    for pattern in sla_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        sla_terms_found.extend(matches)
    
    # Format SLA to match LLM structure with better filtering
    if sla_terms_found:
        performance_metrics = []
        guarantees = []
        
        for term in sla_terms_found:
            term = clean_extracted_text(term)
            
            # Skip if term is too short or just a fragment
            if len(term) < 5 or term.lower() in ['agreements', 'metrics:', 'indicators:', 'sla', 'service']:
                continue
                
            # Check if it contains measurable metrics
            if re.search(r'\d+\s*%|\d+\s*(?:hours?|days?|minutes?)|\d+/\d+|uptime|availability|response\s+time', term, re.IGNORECASE):
                performance_metrics.append(term)
            # Check if it's a meaningful guarantee
            elif len(term) > 10 and re.search(r'guarantee|ensure|provide|maintain|deliver|support', term, re.IGNORECASE):
                guarantees.append(term)
        
        extracted_data["sla"] = {
            "performance_metrics": performance_metrics[:3] if performance_metrics else None,
            "guarantees": guarantees[:3] if guarantees else None
        }
    else:
        extracted_data["sla"] = None
    
    # Extract contact information
    contact_patterns = [
        r'(?:email|e-mail)\s*[:\-]?\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',
        r'(?:phone|tel|telephone)\s*[:\-]?\s*([\d\s\-\(\)\+]{10,20})',
        r'(?:contact|manager|director|officer)\s*[:\-]?\s*([A-Za-z\s]{2,50})',
        r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})',  # Email pattern
        r'(?:address|location)\s*[:\-]?\s*([^\n]{10,100})'
    ]
    
    contacts_found = []
    for pattern in contact_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        contacts_found.extend(matches)
    
    # Format contacts to match LLM structure with better cleaning
    if contacts_found:
        formatted_contacts = []
        for contact in contacts_found[:10]:
            contact = clean_extracted_text(contact)
            
            # Skip if contact is too short or empty after cleaning
            if len(contact) < 3:
                continue
                
            # Determine contact type
            if '@' in contact and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', contact):
                formatted_contacts.append({
                    "type": "email",
                    "value": contact,
                    "name": None
                })
            elif re.match(r'^[\d\s\-\(\)\+]{10,20}$', contact):
                # Clean phone number format
                phone_clean = re.sub(r'[^\d\-\(\)\+]', '', contact)
                if len(phone_clean) >= 10:
                    formatted_contacts.append({
                        "type": "phone",
                        "value": phone_clean,
                        "name": None
                    })
            elif len(contact) > 10 and not '@' in contact and not contact.isdigit():
                # Only include if it looks like a real address
                if re.search(r'\d+.*(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|boulevard|blvd)', contact, re.IGNORECASE):
                    formatted_contacts.append({
                        "type": "address",
                        "value": contact,
                        "name": None
                    })
            elif re.match(r'^[A-Z][a-zA-Z\s]{2,30}$', contact):
                formatted_contacts.append({
                    "type": "person",
                    "value": contact,
                    "name": contact
                })
        
        extracted_data["contacts"] = formatted_contacts
    else:
        extracted_data["contacts"] = []
    
    # Extract dates
    date_patterns = [
        r'(?:date|dated)\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(?:effective|start|begin)\s+(?:date)?\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(?:end|expir|terminat)\s+(?:date)?\s*[:\-]?\s*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})',
        r'(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}'
    ]
    
    dates_found = []
    for pattern in date_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        dates_found.extend(matches)
    
    # Extract contract duration/term
    duration_patterns = [
        r'(?:term|duration|period)\s*[:\-]?\s*(\d+)\s*(?:days?|weeks?|months?|years?)',
        r'for\s+a\s+(?:period\s+of\s+)?(\d+)\s*(?:days?|weeks?|months?|years?)',
        r'(?:valid|effective)\s+for\s+(\d+)\s*(?:days?|weeks?|months?|years?)'
    ]
    
    duration_found = []
    for pattern in duration_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        duration_found.extend(matches)
    
    # Populate additional fields
    additional_fields = {}
    if dates_found:
        additional_fields["dates_found"] = dates_found[:5]
    if duration_found:
        additional_fields["contract_duration"] = duration_found[:3]
    
    # Extract contract type
    contract_type_patterns = [
        r'(?:service|employment|consulting|purchase|sales|lease|rental|license|partnership|joint venture|non-disclosure|confidentiality)\s+(?:agreement|contract)',
        r'(?:agreement|contract)\s+(?:for|of)\s+([^\n\.]{10,50})'
    ]
    
    contract_types = []
    for pattern in contract_type_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        contract_types.extend(matches)
    
    if contract_types:
        additional_fields["contract_type"] = contract_types[0]
    
    # Extract jurisdiction/governing law
    jurisdiction_patterns = [
        r'(?:governed|jurisdiction|laws?)\s+of\s+([A-Za-z\s]{2,30})',
        r'(?:state|province|country)\s+of\s+([A-Za-z\s]{2,30})'
    ]
    
    jurisdictions = []
    for pattern in jurisdiction_patterns:
        matches = re.findall(pattern, contract_text, re.IGNORECASE)
        jurisdictions.extend(matches)
    
    if jurisdictions:
        additional_fields["jurisdiction"] = jurisdictions[0]
    
    # Format additional_fields to match LLM structure with cleaning
    if dates_found:
        cleaned_dates = [clean_extracted_text(date) for date in dates_found[:5]]
        cleaned_dates = [date for date in cleaned_dates if len(date) > 5]  # Filter out fragments
        if cleaned_dates:
            additional_fields["dates_found"] = cleaned_dates
    
    if duration_found:
        cleaned_duration = [clean_extracted_text(dur) for dur in duration_found[:3]]
        if cleaned_duration:
            additional_fields["contract_duration"] = cleaned_duration
    
    if contract_types:
        contract_type = clean_extracted_text(contract_types[0])
        if len(contract_type) > 5:  # Ensure it's not just a fragment
            additional_fields["contract_type"] = contract_type
    
    if jurisdictions:
        jurisdiction = clean_extracted_text(jurisdictions[0])
        if len(jurisdiction) > 5:  # Ensure it's not just a fragment
            additional_fields["jurisdiction"] = jurisdiction
    
    # Add extraction metadata
    additional_fields["extraction_method"] = "regex_fallback"
    additional_fields["extraction_timestamp"] = datetime.now().isoformat()
    
    extracted_data["additional_fields"] = additional_fields
    
    # Ensure null values for empty fields to match LLM format
    for key in extracted_data:
        if key != "additional_fields" and not extracted_data[key]:
            extracted_data[key] = None
    
    print(f"✅ Regex extraction completed. Found {len(extracted_data['parties']) if extracted_data['parties'] else 0} parties, {len(amounts_found)} financial amounts")
    print(f"📊 Extraction quality: Parties={len(extracted_data['parties']) if extracted_data['parties'] else 0}, Contacts={len(extracted_data['contacts']) if extracted_data['contacts'] else 0}, SLA={'Yes' if extracted_data['sla'] else 'No'}")
    
    return extracted_data

def combine_chunk_results(chunk_results: list) -> dict:
    """Combine results from multiple chunks into a single coherent result"""
    if not chunk_results:
        return {}
    
    # Initialize with empty structures
    combined = {
        "parties": [],
        "financials": {},
        "payment_terms": {},
        "sla": {},
        "contacts": [],
        "additional_fields": {}
    }
    
    # Track seen items to avoid duplicates
    seen_parties = set()
    seen_contacts = set()
    
    for result in chunk_results:
        # Merge parties (avoiding duplicates)
        if result.get("parties"):
            for party in result["parties"]:
                # Create a simple hash for deduplication
                if isinstance(party, dict) and party.get("name"):
                    party_key = party["name"].lower()
                    if party_key not in seen_parties:
                        seen_parties.add(party_key)
                        combined["parties"].append(party)
                elif isinstance(party, str) and party.lower() not in seen_parties:
                    seen_parties.add(party.lower())
                    combined["parties"].append(party)
        
        # Merge financials (take most complete information)
        if result.get("financials") and isinstance(result["financials"], dict):
            combined["financials"].update(result["financials"])
        
        # Merge payment terms
        if result.get("payment_terms") and isinstance(result["payment_terms"], dict):
            combined["payment_terms"].update(result["payment_terms"])
        
        # Merge SLA information
        if result.get("sla") and isinstance(result["sla"], dict):
            combined["sla"].update(result["sla"])
        
        # Merge contacts (avoiding duplicates)
        if result.get("contacts"):
            for contact in result["contacts"]:
                # Create a simple hash for deduplication
                if isinstance(contact, dict) and contact.get("name"):
                    contact_key = contact["name"].lower()
                    if contact_key not in seen_contacts:
                        seen_contacts.add(contact_key)
                        combined["contacts"].append(contact)
                elif isinstance(contact, str) and contact.lower() not in seen_contacts:
                    seen_contacts.add(contact.lower())
                    combined["contacts"].append(contact)
        
        # Merge additional fields
        if result.get("additional_fields") and isinstance(result["additional_fields"], dict):
            combined["additional_fields"].update(result["additional_fields"])
    
    return combined

async def query_mistral_llm_single(contract_text: str, is_chunk=False, chunk_num=None, total_chunks=None) -> dict:
    """Query the Mistral LLM for a single chunk of text"""
    system_prompt = "You are a specialized AI assistant for contract analysis. Your task is to extract structured information from contract documents and return it as valid JSON. Be precise and thorough in your extraction."
    
    # Adjust prompt based on whether this is a chunk or full document
    if is_chunk:
        user_prompt = f"""You are analyzing chunk {chunk_num} of {total_chunks} from a large contract document.

Extract the following contract details as a valid JSON object with these keys:

1. parties: Array of entities involved in the contract (names, roles, addresses)
2. financials: Object containing monetary details (amounts, currencies, total value)
3. payment_terms: Object with payment schedule, methods, penalties
4. sla: Service level agreements, performance metrics, guarantees
5. contacts: Key personnel, points of contact, contact information
6. additional_fields: Object containing ANY other important information found in the contract

If any field is missing or cannot be determined, set it as null or an empty structure. Focus on extracting what's available in this chunk.

Your response MUST be a valid JSON object without any explanatory text before or after.

Contract text (chunk {chunk_num}/{total_chunks}):
{contract_text}
"""
    else:
        user_prompt = f"""Extract the following contract details as a valid JSON object with these keys:

1. parties: Array of entities involved in the contract (names, roles, addresses)
2. financials: Object containing monetary details (amounts, currencies, total value)
3. payment_terms: Object with payment schedule, methods, penalties
4. sla: Service level agreements, performance metrics, guarantees
5. contacts: Key personnel, points of contact, contact information
6. additional_fields: Object containing ANY other important information found in the contract

If any field is missing or cannot be determined, set it as null. For additional_fields, include any important information that doesn't fit the standard categories.

Your response MUST be a valid JSON object without any explanatory text before or after. Format:
{{"parties": [...], "financials": {{...}}, "payment_terms": {{...}}, "sla": {{...}}, "contacts": {{...}}, "additional_fields": {{...}}}}

Contract text:
{contract_text}
"""

    try:
        # First try with httpx
        # Use proper SSL verification with system certificates
        async with httpx.AsyncClient(timeout=60, verify=certifi.where()) as client:
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    text_output = data["choices"][0]["message"]["content"]
                    
                    try:
                        # First try direct JSON parsing
                        extracted = json.loads(text_output)
                        return extracted
                    except json.JSONDecodeError:
                        print("JSON parsing failed, attempting to extract JSON from text")
                        # Try to extract JSON from text (sometimes model wraps JSON in markdown or explanations)
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', text_output)
                        if json_match:
                            try:
                                json_str = json_match.group(0)
                                # Try to parse as is
                                try:
                                    extracted = json.loads(json_str)
                                    return extracted
                                except json.JSONDecodeError:
                                    # Try to fix truncated JSON by adding missing closing braces
                                    print("Attempting to fix truncated JSON")
                                    # Count opening and closing braces
                                    open_braces = json_str.count('{')
                                    close_braces = json_str.count('}')
                                    if open_braces > close_braces:
                                        # Add missing closing braces
                                        json_str += '}' * (open_braces - close_braces)
                                        try:
                                            extracted = json.loads(json_str)
                                            return extracted
                                        except json.JSONDecodeError:
                                            pass
                            except Exception as e:
                                print(f"Error extracting JSON: {str(e)}")
            
            # If we get here, either the API call failed or JSON parsing failed
            print(f"OpenRouter API call failed with status {response.status_code}")
            print(f"Response: {response.text}")
            
            # Try the fallback method
            return await query_mistral_llm_requests_fallback(contract_text)
            
    except Exception as e:
        print(f"Error in query_mistral_llm: {str(e)}")
        # Try the fallback method
        return await query_mistral_llm_requests_fallback(contract_text)

async def query_mistral_llm_requests_fallback(contract_text: str) -> dict:
    """
    Fallback function using requests library with aggressive SSL bypass
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
    system_prompt = "You are a specialized AI assistant for contract analysis. Your task is to extract structured information from contract documents and return it as valid JSON. Be precise and thorough in your extraction."
    
    user_prompt = f"""Extract the following contract details as a valid JSON object with these keys:

1. parties: Array of entities involved in the contract (names, roles, addresses)
2. financials: Object containing monetary details (amounts, currencies, total value)
3. payment_terms: Object with payment schedule, methods, penalties
4. sla: Service level agreements, performance metrics, guarantees
5. contacts: Key personnel, points of contact, contact information
6. additional_fields: Object containing ANY other important information found in the contract

If any field is missing or cannot be determined, set it as null. For additional_fields, include any important information that doesn't fit the standard categories.

Your response MUST be a valid JSON object without any explanatory text before or after. Format:
{{"parties": [...], "financials": {{...}}, "payment_terms": {{...}}, "sla": {{...}}, "contacts": {{...}}, "additional_fields": {{...}}}}

Contract text:
{contract_text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": "ContractIntelligence/1.0"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    # Configure requests session with proper SSL verification
    session = requests.Session()
    session.verify = certifi.where()  # Use system certificates for SSL verification
    
    # Add retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Use default SSL context with proper verification
    # No need to create a custom SSL context that bypasses security
    
    endpoints = [
        "https://openrouter.ai/api/v1/chat/completions",
        "https://api.openrouter.ai/api/v1/chat/completions"
    ]
    
    for endpoint in endpoints:
        try:
            print(f"Requests fallback: trying {endpoint}")
            response = session.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=60,
                verify=certifi.where()
            )
            
            print(f"Requests response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    text_output = data["choices"][0]["message"]["content"]
                    print("✅ Requests fallback successful!")
                    
                    try:
                        # First try direct JSON parsing
                        extracted = json.loads(text_output)
                        return extracted
                    except json.JSONDecodeError:
                        print("JSON parsing failed, attempting to extract JSON from text")
                        # Try to extract JSON from text (sometimes model wraps JSON in markdown or explanations)
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', text_output)
                        if json_match:
                            try:
                                json_str = json_match.group(0)
                                # Try to parse as is
                                try:
                                    extracted = json.loads(json_str)
                                    return extracted
                                except json.JSONDecodeError:
                                    # Try to fix truncated JSON by adding missing closing braces
                                    print("Attempting to fix truncated JSON")
                                    # Count opening and closing braces
                                    open_braces = json_str.count('{')
                                    close_braces = json_str.count('}')
                                    open_brackets = json_str.count('[')
                                    close_brackets = json_str.count(']')
                                    
                                    # Add missing closing braces and brackets
                                    if open_braces > close_braces:
                                        json_str += '}' * (open_braces - close_braces)
                                    if open_brackets > close_brackets:
                                        json_str += ']' * (open_brackets - close_brackets)
                                        
                                    try:
                                        extracted = json.loads(json_str)
                                        print("Successfully fixed truncated JSON")
                                        return extracted
                                    except json.JSONDecodeError:
                                        print("Could not fix truncated JSON")
                            except Exception as e:
                                print(f"Error processing JSON match: {str(e)}")                          
                        # Try to extract JSON from raw text using a more aggressive approach
                        try:
                            # Look for JSON-like structure and try to parse it
                            # First, try to find the raw JSON structure
                            raw_json = text_output
                            
                            # Try to extract the JSON content from the raw text
                            # This is a more aggressive approach to find valid JSON
                            import re
                            # Find all JSON-like structures
                            json_candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_json)
                            
                            for candidate in json_candidates:
                                try:
                                    # Try to parse each candidate
                                    parsed = json.loads(candidate)
                                    if isinstance(parsed, dict) and any(key in parsed for key in ['parties', 'financials', 'payment_terms', 'sla', 'contacts']):
                                        print("Found valid JSON structure in raw text")
                                        return parsed
                                except:
                                    continue
                        except Exception as e:
                            print(f"Advanced JSON extraction failed: {str(e)}")
                            
                        # If all parsing fails, try to extract structured data from the raw text
                        try:
                            # Try to parse the raw text as JSON with missing closing braces
                            raw_text = text_output
                            # Find the start of a JSON object
                            json_start = raw_text.find('{')
                            if json_start >= 0:
                                partial_json = raw_text[json_start:]
                                # Count opening and closing braces
                                open_braces = partial_json.count('{')
                                close_braces = partial_json.count('}')
                                open_brackets = partial_json.count('[')
                                close_brackets = partial_json.count(']')
                                
                                # Add missing closing braces and brackets
                                if open_braces > close_braces:
                                    partial_json += '}' * (open_braces - close_braces)
                                if open_brackets > close_brackets:
                                    partial_json += ']' * (open_brackets - close_brackets)
                                    
                                try:
                                    fixed_json = json.loads(partial_json)
                                    print("Successfully parsed fixed JSON")
                                    return fixed_json
                                except:
                                    pass
                        except Exception as e:
                            print(f"Failed to fix JSON: {str(e)}")
                            
                        # If all parsing fails, return structured data with the raw text
                        return {
                            "parties": None, 
                            "financials": None, 
                            "payment_terms": None, 
                            "sla": None, 
                            "contacts": None,
                            "additional_fields": {
                                "raw_text": text_output
                            }
                        }
            else:
                print(f"Requests error: {response.status_code} - {response.text}")
                
        except Exception as e:
            print(f"Requests exception for {endpoint}: {str(e)}")
            continue
    
    # If all requests attempts fail, use regex fallback
    print("🔄 All requests fallback attempts failed, using regex extraction")
    return extract_contract_data_with_regex(contract_text)

async def query_mistral_llm(contract_text: str) -> dict:
    """
    Send contract text to Mistral model via OpenRouter API with comprehensive fallbacks.
    Returns structured JSON fields.
    """
    # Check if API key is available
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set. Please set it before making API calls.")
        
    system_prompt = "You are a specialized AI assistant for contract analysis. Your task is to extract structured information from contract documents and return it as valid JSON. Be precise and thorough in your extraction."
    
    user_prompt = f"""Extract the following contract details as a valid JSON object with these keys:

1. parties: Array of entities involved in the contract (names, roles, addresses)
2. financials: Object containing monetary details (amounts, currencies, total value)
3. payment_terms: Object with payment schedule, methods, penalties
4. sla: Service level agreements, performance metrics, guarantees
5. contacts: Key personnel, points of contact, contact information
6. additional_fields: Object containing ANY other important information found in the contract

If any field is missing or cannot be determined, set it as null. For additional_fields, include any important information that doesn't fit the standard categories.

Your response MUST be a valid JSON object without any explanatory text before or after. Format:
{{"parties": [...], "financials": {{...}}, "payment_terms": {{...}}, "sla": {{...}}, "contacts": {{...}}, "additional_fields": {{...}}}}

Contract text:
{contract_text}
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "Contract Intelligence API",
        "User-Agent": "ContractIntelligence/1.0"
    }
    
    payload = {
        "model": MISTRAL_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 1000
    }

    # Multiple endpoint strategies with DNS resolution fallbacks
    endpoint_strategies = [
        # Strategy 1: Standard hostnames
        {
            "name": "Standard OpenRouter endpoint",
            "url": "https://openrouter.ai/api/v1/chat/completions"
        },
        {
            "name": "Alternative OpenRouter endpoint", 
            "url": "https://api.openrouter.ai/api/v1/chat/completions"
        },
        # Strategy 2: Direct IP addresses (bypassing DNS)
        {
            "name": "Direct IP for openrouter.ai",
            "url": "https://104.18.6.192/api/v1/chat/completions",
            "headers": {**headers, "Host": "openrouter.ai"}
        },
        {
            "name": "Direct IP for api.openrouter.ai",
            "url": "https://104.18.7.192/api/v1/chat/completions", 
            "headers": {**headers, "Host": "api.openrouter.ai"}
        }
    ]
    
    # Different client configurations to try
    client_configs = [
        {
            "name": "Standard secure with certificates",
            "timeout": httpx.Timeout(60.0, connect=30.0),
            "verify": certifi.where(),
            "follow_redirects": True
        },
        {
            "name": "Extended timeout with certificates",
            "timeout": httpx.Timeout(120.0, connect=60.0),
            "verify": certifi.where(),
            "follow_redirects": True
        },
        {
            "name": "System SSL context",
            "timeout": httpx.Timeout(60.0, connect=30.0),
            "verify": ssl_context,
            "follow_redirects": True
        },
        {
            "name": "Basic secure connection",
            "timeout": httpx.Timeout(50.0, connect=25.0),
            "verify": certifi.where(),
            "follow_redirects": False
        }
    ]
    
    last_error = None
    
    for strategy in endpoint_strategies:
        endpoint = strategy["url"]
        endpoint_headers = strategy.get("headers", headers)
        
        print(f"Trying strategy: {strategy['name']} - {endpoint}")
        
        for config in client_configs:
            try:
                print(f"  Using config: {config['name']}")
                
                async with httpx.AsyncClient(**{k: v for k, v in config.items() if k != 'name'}) as client:
                    try:
                        response = await client.post(
                            endpoint,
                            json=payload,
                            headers=endpoint_headers
                        )
                        
                        print(f"  Response status: {response.status_code}")
                        response.raise_for_status()
                        data = response.json()
                        
                        # Parse response according to the OpenRouter API format
                        if "choices" in data and len(data["choices"]) > 0:
                            text_output = data["choices"][0]["message"]["content"]
                            print(f"✅ Success with {strategy['name']}")
                            
                            # Try to parse JSON from response
                            try:
                                # First try direct JSON parsing
                                extracted = json.loads(text_output)
                                return extracted
                            except json.JSONDecodeError:
                                print("Warning: Could not parse LLM response as JSON, attempting to extract JSON from text")
                                # Try to extract JSON from text (sometimes model wraps JSON in markdown or explanations)
                                import re
                                json_match = re.search(r'\{[\s\S]*\}', text_output)
                                if json_match:
                                    try:
                                        json_str = json_match.group(0)
                                        # Try to parse as is
                                        try:
                                            extracted = json.loads(json_str)
                                            return extracted
                                        except json.JSONDecodeError:
                                            # Try to fix truncated JSON by adding missing closing braces
                                            print("Attempting to fix truncated JSON")
                                            # Count opening and closing braces
                                            open_braces = json_str.count('{')
                                            close_braces = json_str.count('}')
                                            open_brackets = json_str.count('[')
                                            close_brackets = json_str.count(']')
                                            
                                            # Add missing closing braces and brackets
                                            if open_braces > close_braces:
                                                json_str += '}' * (open_braces - close_braces)
                                            if open_brackets > close_brackets:
                                                json_str += ']' * (open_brackets - close_brackets)
                                                
                                            try:
                                                extracted = json.loads(json_str)
                                                print("Successfully fixed truncated JSON")
                                                return extracted
                                            except json.JSONDecodeError:
                                                print("Could not fix truncated JSON")
                                    except Exception as e:
                                        print(f"Error processing JSON match: {str(e)}")
                                        
                                # Try to extract JSON from raw text using a more aggressive approach
                                try:
                                    # Look for JSON-like structure and try to parse it
                                    # First, try to find the raw JSON structure
                                    raw_json = text_output
                                    
                                    # Try to extract the JSON content from the raw text
                                    # This is a more aggressive approach to find valid JSON
                                    import re
                                    # Find all JSON-like structures
                                    json_candidates = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_json)
                                    
                                    for candidate in json_candidates:
                                        try:
                                            # Try to parse each candidate
                                            parsed = json.loads(candidate)
                                            if isinstance(parsed, dict) and any(key in parsed for key in ['parties', 'financials', 'payment_terms', 'sla', 'contacts']):
                                                print("Found valid JSON structure in raw text")
                                                return parsed
                                        except:
                                            continue
                                except Exception as e:
                                    print(f"Advanced JSON extraction failed: {str(e)}")
                                    
                                # If all parsing fails, try to extract structured data from the raw text
                                try:
                                    # Try to parse the raw text as JSON with missing closing braces
                                    raw_text = text_output
                                    # Find the start of a JSON object
                                    json_start = raw_text.find('{')
                                    if json_start >= 0:
                                        partial_json = raw_text[json_start:]
                                        # Count opening and closing braces
                                        open_braces = partial_json.count('{')
                                        close_braces = partial_json.count('}')
                                        open_brackets = partial_json.count('[')
                                        close_brackets = partial_json.count(']')
                                        
                                        # Add missing closing braces and brackets
                                        if open_braces > close_braces:
                                            partial_json += '}' * (open_braces - close_braces)
                                        if open_brackets > close_brackets:
                                            partial_json += ']' * (open_brackets - close_brackets)
                                            
                                        try:
                                            fixed_json = json.loads(partial_json)
                                            print("Successfully parsed fixed JSON")
                                            return fixed_json
                                        except:
                                            pass
                                except Exception as e:
                                    print(f"Failed to fix JSON: {str(e)}")
                                    
                                # If all parsing fails, fall back to regex extraction
                                print("🔄 JSON parsing completely failed, falling back to regex extraction")
                                return extract_contract_data_with_regex(contract_text)
                        else:
                            last_error = ValueError(f"Unexpected API response format: {data}")
                            
                    except httpx.HTTPStatusError as e:
                        last_error = ValueError(f"HTTP {e.response.status_code}: {e.response.text}")
                        print(f"  HTTP Error: {e.response.status_code}")
                    except httpx.ConnectError as e:
                        last_error = ValueError(f"Connection error: {str(e)}")
                        print(f"  Connection Error: {str(e)}")
                    except httpx.TimeoutException as e:
                        last_error = ValueError(f"Timeout error: {str(e)}")
                        print(f"  Timeout Error: {str(e)}")
                    except Exception as e:
                        last_error = ValueError(f"Request error: {str(e)}")
                        print(f"  Request Error: {str(e)}")
                        
            except Exception as e:
                last_error = ValueError(f"Client creation error: {str(e)}")
                print(f"  Client Error: {str(e)}")
    
    # If all httpx attempts failed, try requests fallback
    if last_error:
        print("🔄 Trying requests library fallback...")
        try:
            return await query_mistral_llm_requests_fallback(contract_text)
        except Exception as fallback_error:
            print(f"Requests fallback also failed: {str(fallback_error)}")
            
            # Provide a helpful error message based on the original httpx errors
            error_msg = str(last_error)
            if "getaddrinfo failed" in error_msg or "Name or service not known" in error_msg:
                print("🔄 DNS resolution failed, using regex fallback extraction")
                return extract_contract_data_with_regex(contract_text)
            elif "SSL" in error_msg.upper() or "certificate" in error_msg.lower() or "handshake" in error_msg.lower():
                print("🔄 SSL connection failed, using regex fallback extraction")
                return extract_contract_data_with_regex(contract_text)
            elif "timeout" in error_msg.lower():
                print("🔄 Connection timeout, using regex fallback extraction")
                return extract_contract_data_with_regex(contract_text)
            else:
                print(f"🔄 All API connection attempts failed, using regex fallback extraction")
                return extract_contract_data_with_regex(contract_text)
    
    # This should never be reached, but if it does, use regex fallback
    print("🔄 Unexpected API failure, using regex fallback extraction")
    return extract_contract_data_with_regex(contract_text)

# Mock data generation removed as per requirement

async def process_contract(contract_id: str, file_path: str):
    """Process a contract file and extract information with memory optimization"""
    global USE_MONGO
    
    try:
        # Update status to processing
        await update_contract_status(contract_id, status="processing", progress=5)
        
        # Get file size for progress tracking
        file_size = os.path.getsize(file_path)
        is_large_file = file_size > 10 * 1024 * 1024  # 10MB threshold
        
        if is_large_file:
            print(f"Processing large PDF ({file_size / (1024 * 1024):.2f} MB) with optimized memory usage")
        
        # Extract text from PDF with memory optimization
        contract_text = await extract_text_from_pdf(file_path)
        
        # Free up memory by clearing references
        import gc
        gc.collect()
        
        await update_contract_status(contract_id, status="processing", progress=30)
        
        # Check internet connectivity before proceeding (but don't fail if no connectivity)
        has_connectivity = await test_network_connectivity()
        if not has_connectivity:
            print(f"No internet connectivity detected for contract {contract_id}, will use regex fallback")
        
        await update_contract_status(contract_id, status="processing", progress=50)
        
        # Query LLM for extraction with chunking for large documents
        # This will automatically fall back to regex extraction if API fails
        extracted_data = await query_mistral_llm(contract_text)
        
        # Free up memory again
        contract_text = None
        gc.collect()
        
        await update_contract_status(contract_id, status="processing", progress=90)
        
        # Update contract with extracted data
        await update_contract_status(contract_id, status="completed", progress=100)
        await update_contract(contract_id, {
            "extracted_data": extracted_data
        })
    except Exception as e:
        error_message = str(e)
        # Provide more user-friendly error messages
        if "getaddrinfo failed" in error_message:
            error_message = "DNS resolution failed. Please check your internet connection and DNS settings."
        elif "SSL" in error_message.upper() or "certificate" in error_message.lower():
            error_message = "SSL connection failed. This might be due to firewall or network security settings. Please check your network configuration."
        elif "timeout" in error_message.lower():
            error_message = "Connection timeout. Please check your internet connection."
        
        await update_contract_status(contract_id, status="error", progress=100, error=error_message)

# ----------------- Routes -----------------
@app.post("/contracts/upload")
async def upload_contract(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contract_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{contract_id}.pdf")
    
    # Stream the file to disk in chunks to handle large files
    file_size = 0
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        with open(file_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                file_size += len(chunk)
                f.write(chunk)
                
        print(f"Uploaded file size: {file_size / (1024 * 1024):.2f} MB")
    except Exception as e:
        print(f"Error during file upload: {str(e)}")
        if os.path.exists(file_path):
            os.remove(file_path)  # Clean up partial file
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

    # Create contract document
    contract_doc = {
        "_id": contract_id,
        "filename": file.filename,
        "status": "uploaded",
        "progress": 0,
        "file_size_bytes": file_size,
        "extracted_data": None,
        "error": None,
        "created_at": int(time.time())  # Unix timestamp for sorting
    }
    
    # Store contract metadata based on available storage
    global USE_MONGO
    if USE_MONGO:
        try:
            # Set a short timeout for this operation
            await asyncio.wait_for(
                contracts_collection.insert_one(contract_doc),
                timeout=2.0
            )
        except (asyncio.TimeoutError, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection timed out: {str(e)}")
            print("Switching to file-based storage for future operations")
            USE_MONGO = False
            # Fall back to file storage
            save_contract_to_file(contract_id, contract_doc)
        except Exception as e:
            print(f"MongoDB insert failed: {str(e)}")
            # Fall back to file storage if MongoDB insert fails
            save_contract_to_file(contract_id, contract_doc)
    else:
        # Use file-based storage
        save_contract_to_file(contract_id, contract_doc)

    # Launch background processing
    background_tasks.add_task(process_contract, contract_id, file_path)

    return {"contract_id": contract_id}

@app.get("/contracts/{contract_id}/status", response_model=ContractStatus)
async def get_contract_status(contract_id: str):
    global USE_MONGO
    contract = None
    
    # Try to get contract from available storage
    if USE_MONGO:
        try:
            # Set a short timeout for this operation
            contract = await asyncio.wait_for(
                contracts_collection.find_one({"_id": contract_id}),
                timeout=2.0
            )
        except (asyncio.TimeoutError, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection timed out: {str(e)}")
            print("Switching to file-based storage for future operations")
            USE_MONGO = False
            # Fall back to file storage
            contract = load_contract_from_file(contract_id)
        except Exception as e:
            print(f"MongoDB find_one failed: {str(e)}")
            # Fall back to file storage
            contract = load_contract_from_file(contract_id)
    else:
        # Use file-based storage
        contract = load_contract_from_file(contract_id)
        
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
        
    return ContractStatus(
        status=contract.get("status"),
        progress=contract.get("progress"),
        score=None,
        error=contract.get("error")
    )

@app.get("/contracts/{contract_id}")
async def get_contract_data(contract_id: str):
    global USE_MONGO
    contract = None
    
    # Try to get contract from available storage
    if USE_MONGO:
        try:
            # Set a short timeout for this operation
            contract = await asyncio.wait_for(
                contracts_collection.find_one({"_id": contract_id}),
                timeout=2.0
            )
        except (asyncio.TimeoutError, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection timed out: {str(e)}")
            print("Switching to file-based storage for future operations")
            USE_MONGO = False
            # Fall back to file storage
            contract = load_contract_from_file(contract_id)
        except Exception as e:
            print(f"MongoDB find_one failed: {str(e)}")
            # Fall back to file storage
            contract = load_contract_from_file(contract_id)
    else:
        # Use file-based storage
        contract = load_contract_from_file(contract_id)
        
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
        
    # Return the full contract object instead of just extracted_data
    # This allows the frontend to display contract info for any status
    return contract

@app.get("/contracts/{contract_id}/download")
async def download_contract_pdf(contract_id: str):
    """Download the original PDF file for a contract"""
    # Check if contract exists
    global USE_MONGO
    contract = None
    
    # Try to get contract from available storage to verify it exists
    if USE_MONGO:
        try:
            contract = await asyncio.wait_for(
                contracts_collection.find_one({"_id": contract_id}),
                timeout=2.0
            )
        except Exception:
            # Fall back to file storage
            contract = load_contract_from_file(contract_id)
    else:
        # Use file-based storage
        contract = load_contract_from_file(contract_id)
        
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    
    # Construct the path to the PDF file
    file_path = os.path.join(UPLOAD_DIR, f"{contract_id}.pdf")
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    # Get original filename from contract metadata
    filename = contract.get("filename", f"contract_{contract_id}.pdf")
    
    # Return the file as a downloadable response
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/pdf"
    )

@app.get("/contracts")
async def list_contracts():
    """Get a list of all contracts with their status information"""
    global USE_MONGO
    contracts = []
    
    if USE_MONGO:
        try:
            # Set a short timeout for this operation
            cursor = contracts_collection.find({}, {
                "_id": 1,
                "filename": 1,
                "status": 1,
                "progress": 1,
                "error": 1,
                "created_at": 1
            })
            
            # Use a timeout for the cursor iteration
            async def get_docs_with_timeout():
                docs = []
                async for doc in cursor:
                    # Convert MongoDB _id to contract_id string for frontend
                    # Handle ObjectId serialization
                    if isinstance(doc["_id"], ObjectId):
                        doc["contract_id"] = str(doc["_id"])
                    else:
                        doc["contract_id"] = doc["_id"]
                    
                    # Remove the _id field to avoid serialization issues
                    doc.pop("_id")
                    docs.append(doc)
                return docs
                
            # Execute with timeout
            contracts = await asyncio.wait_for(get_docs_with_timeout(), timeout=3.0)
            
        except (asyncio.TimeoutError, ServerSelectionTimeoutError) as e:
            print(f"MongoDB connection timed out: {str(e)}")
            print("Switching to file-based storage for future operations")
            USE_MONGO = False
            # Fall back to file storage
            contracts = load_all_contracts_from_files()
        except Exception as e:
            print(f"MongoDB find failed: {str(e)}")
            # Fall back to file storage
            contracts = load_all_contracts_from_files()
    else:
        # Use file-based storage
        contracts = load_all_contracts_from_files()
    
    # Sort by most recently created first (if created_at exists)
    contracts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return contracts


# ----------------- File-based Storage Functions -----------------
def save_contract_to_file(contract_id: str, contract_data: dict):
    """Save contract data to a JSON file"""
    try:
        file_path = os.path.join(FILE_STORAGE_DIR, f"{contract_id}.json")
        with open(file_path, 'w') as f:
            json.dump(contract_data, f)
        print(f"✅ Saved contract {contract_id} to file storage")
        return True
    except Exception as e:
        print(f"❌ Error saving contract to file: {str(e)}")
        return False

def load_contract_from_file(contract_id: str) -> dict:
    """Load contract data from a JSON file"""
    try:
        file_path = os.path.join(FILE_STORAGE_DIR, f"{contract_id}.json")
        if not os.path.exists(file_path):
            return None
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading contract from file: {str(e)}")
        return None

def load_all_contracts_from_files() -> list:
    """Load all contracts from JSON files"""
    contracts = []
    try:
        for filename in os.listdir(FILE_STORAGE_DIR):
            if filename.endswith('.json'):
                contract_id = filename.replace('.json', '')
                contract = load_contract_from_file(contract_id)
                if contract:
                    # Add contract_id field for frontend compatibility
                    contract['contract_id'] = contract['_id']
                    # Remove _id to avoid serialization issues
                    if '_id' in contract:
                        contract.pop('_id')
                    contracts.append(contract)
        return contracts
    except Exception as e:
        print(f"❌ Error loading contracts from files: {str(e)}")
        return []

async def update_contract_status(contract_id: str, status=None, progress=None, error=None):
    """Update contract status in the appropriate storage
    
    Can be called with either individual parameters or as a dictionary
    """
    # Create update data dictionary from parameters
    update_data = {}
    if isinstance(status, str):
        update_data["status"] = status
    if progress is not None:
        update_data["progress"] = progress
    if error is not None:
        update_data["error"] = error
        
    # Call the update function with the dictionary
    await update_contract(contract_id, update_data)

async def update_contract(contract_id: str, update_data: dict):
    """Update contract in the appropriate storage"""
    if USE_MONGO:
        try:
            # Try MongoDB first
            await _update_mongo_contract(contract_id, update_data)
        except Exception as e:
            print(f"MongoDB update failed: {str(e)}")
            # Fall back to file storage
            _update_file_contract(contract_id, update_data)
    else:
        # Use file-based storage
        _update_file_contract(contract_id, update_data)

async def _update_mongo_contract(contract_id: str, update_data: dict):
    """Update contract in MongoDB"""
    global USE_MONGO
    try:
        # Set a short timeout for this operation
        await asyncio.wait_for(
            contracts_collection.update_one(
                {"_id": contract_id},
                {"$set": update_data}
            ),
            timeout=2.0
        )
    except (asyncio.TimeoutError, ServerSelectionTimeoutError) as e:
        print(f"MongoDB connection timed out: {str(e)}")
        print("Switching to file-based storage for future operations")
        USE_MONGO = False
        # Fall back to file storage
        _update_file_contract(contract_id, update_data)
    except Exception as e:
        print(f"MongoDB update failed: {str(e)}")
        # Fall back to file storage
        _update_file_contract(contract_id, update_data)

def _update_file_contract(contract_id: str, update_data: dict):
    """Update contract in file storage"""
    try:
        contract = load_contract_from_file(contract_id)
        if contract:
            contract.update(update_data)
            save_contract_to_file(contract_id, contract)
    except Exception as e:
        print(f"File storage update failed: {str(e)}")
