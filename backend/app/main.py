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
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from pydantic import BaseModel
import motor.motor_asyncio
import httpx
from bson import ObjectId

# Disable SSL warnings for requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables from .env file if it exists
env_path = pathlib.Path(__file__).parent.parent / '.env'
if env_path.exists():
    print(f"Loading environment variables from {env_path}")
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ[key] = value

# ----------------- Configuration -----------------
MONGO_URL = "mongodb://localhost:27017"
DB_NAME = "contracts_db"
COLLECTION_NAME = "contracts"

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")  # set in .env
if not OPENROUTER_API_KEY:
    print("WARNING: OPENROUTER_API_KEY environment variable is not set. API calls will fail.")
MISTRAL_MODEL = "mistralai/mistral-7b-instruct:free"  # free model

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ----------------- MongoDB Client -----------------
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]
contracts_collection = db[COLLECTION_NAME]

# ----------------- FastAPI App -----------------
app = FastAPI(title="Contract Intelligence API")

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
    score: int | None = None
    error: str | None = None

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
        "https://8.8.8.8",  # Google DNS
        "https://1.1.1.1"   # Cloudflare DNS
    ]
    
    for url in test_urls:
        try:
            async with httpx.AsyncClient(timeout=10, verify=False) as client:
                response = await client.get(url)
                if response.status_code in [200, 301, 302]:
                    return True
        except Exception:
            continue
    return False

async def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

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

    # Configure requests session with SSL bypass
    session = requests.Session()
    session.verify = False  # Disable SSL verification completely
    
    # Add retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Create custom SSL context that's more permissive
    import ssl
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
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
                verify=False
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
    
    raise ValueError("All requests fallback attempts failed")

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
            "name": "Standard secure",
            "timeout": httpx.Timeout(60.0, connect=30.0),
            "verify": True,
            "follow_redirects": True
        },
        {
            "name": "Extended timeout",
            "timeout": httpx.Timeout(120.0, connect=60.0),
            "verify": True,
            "follow_redirects": True
        },
        {
            "name": "No SSL verification",
            "timeout": httpx.Timeout(60.0, connect=30.0),
            "verify": False,
            "follow_redirects": True
        },
        {
            "name": "Basic connection",
            "timeout": httpx.Timeout(30.0, connect=15.0),
            "verify": False,
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
                raise ValueError("DNS resolution failed. This could be due to: 1) No internet connection, 2) DNS server issues, 3) Firewall blocking requests. Please check your network settings.")
            elif "SSL" in error_msg.upper() or "certificate" in error_msg.lower() or "handshake" in error_msg.lower():
                raise ValueError("SSL/TLS connection failed. This might be due to corporate firewall or security settings. All SSL bypass attempts failed.")
            elif "timeout" in error_msg.lower():
                raise ValueError("Connection timeout. Please check your internet speed and try again.")
            else:
                raise ValueError(f"All connection attempts failed. HTTPX error: {error_msg}. Requests error: {str(fallback_error)}")
    
    # This should never be reached
    raise ValueError("All API connection attempts failed without specific error")

# Mock data generation removed as per requirement

async def process_contract(contract_id: str, file_path: str):
    """
    Background task: extract PDF text, query LLM, update MongoDB
    """
    try:
        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {"status": "processing", "progress": 10}}
        )

        text = await extract_text_from_pdf(file_path)

        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {"progress": 30}}
        )

        if not OPENROUTER_API_KEY:
            # If API key is missing, don't extract any data
            await contracts_collection.update_one(
                {"_id": contract_id},
                {"$set": {
                    "status": "completed",
                    "progress": 100,
                    "extracted_data": None,
                    "error": "No API key provided. Only PDF storage is available."
                }}
            )
            return

        # Test network connectivity first
        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {"progress": 40}}
        )
        
        network_ok = await test_network_connectivity()
        if not network_ok:
            await contracts_collection.update_one(
                {"_id": contract_id},
                {"$set": {
                    "status": "error",
                    "progress": 100,
                    "error": "No internet connectivity detected. Please check your network connection."
                }}
            )
            return
            
        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {"progress": 50}}
        )
            
        # If API key is available and network is working, proceed with extraction
        extracted_data = await query_mistral_llm(text)

        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {
                "status": "completed",
                "progress": 100,
                "extracted_data": extracted_data
            }}
        )
    except Exception as e:
        error_message = str(e)
        # Provide more user-friendly error messages
        if "getaddrinfo failed" in error_message:
            error_message = "DNS resolution failed. Please check your internet connection and DNS settings."
        elif "SSL" in error_message.upper():
            error_message = "SSL connection failed. This might be due to firewall or network security settings."
        elif "timeout" in error_message.lower():
            error_message = "Connection timeout. Please check your internet connection."
        
        await contracts_collection.update_one(
            {"_id": contract_id},
            {"$set": {
                "status": "error",
                "progress": 100,
                "error": error_message
            }}
        )

# ----------------- Routes -----------------
@app.post("/contracts/upload")
async def upload_contract(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    contract_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{contract_id}.pdf")
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    contract_doc = {
        "_id": contract_id,
        "filename": file.filename,
        "status": "uploaded",
        "progress": 0,
        "extracted_data": None,
        "error": None,
        "created_at": int(time.time())  # Unix timestamp for sorting
    }
    await contracts_collection.insert_one(contract_doc)

    # Launch background processing
    background_tasks.add_task(process_contract, contract_id, file_path)

    return {"contract_id": contract_id}

@app.get("/contracts/{contract_id}/status", response_model=ContractStatus)
async def get_contract_status(contract_id: str):
    contract = await contracts_collection.find_one({"_id": contract_id})
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
    contract = await contracts_collection.find_one({"_id": contract_id})
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return contract.get("extracted_data", {})

@app.get("/contracts")
async def list_contracts():
    """Get a list of all contracts with their status information"""
    contracts = []
    cursor = contracts_collection.find({}, {
        "_id": 1,
        "filename": 1,
        "status": 1,
        "progress": 1,
        "error": 1,
        "created_at": 1
    })
    
    async for doc in cursor:
        # Convert MongoDB _id to contract_id string for frontend
        # Handle ObjectId serialization
        if isinstance(doc["_id"], ObjectId):
            doc["contract_id"] = str(doc["_id"])
        else:
            doc["contract_id"] = doc["_id"]
        
        # Remove the _id field to avoid serialization issues
        doc.pop("_id")
        contracts.append(doc)
    
    # Sort by most recently created first (if created_at exists)
    contracts.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    
    return contracts

