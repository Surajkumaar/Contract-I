#!/usr/bin/env python3
"""
Test script to verify SSL connection fixes for OpenRouter API
"""
import asyncio
import httpx
import requests
import ssl
import certifi
import os
import sys

# Add the backend app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend', 'app'))

async def test_ssl_connections():
    """Test various SSL connection methods to OpenRouter API"""
    
    print("🔍 Testing SSL connections to OpenRouter API...")
    print("=" * 50)
    
    # Test URLs
    test_endpoints = [
        "https://openrouter.ai/api/v1/models",
        "https://api.openrouter.ai/api/v1/models"
    ]
    
    # Test configurations
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    ssl_context.load_default_certs()
    
    configs = [
        {
            "name": "HTTPX with certifi certificates",
            "client": lambda: httpx.AsyncClient(timeout=30, verify=certifi.where())
        },
        {
            "name": "HTTPX with system SSL context", 
            "client": lambda: httpx.AsyncClient(timeout=30, verify=ssl_context)
        },
        {
            "name": "HTTPX with default verification",
            "client": lambda: httpx.AsyncClient(timeout=30, verify=True)
        }
    ]
    
    for endpoint in test_endpoints:
        print(f"\n📡 Testing endpoint: {endpoint}")
        print("-" * 40)
        
        for config in configs:
            try:
                print(f"  Testing: {config['name']}")
                async with config["client"]() as client:
                    response = await client.get(endpoint)
                    if response.status_code == 200:
                        print(f"  ✅ SUCCESS - Status: {response.status_code}")
                    else:
                        print(f"  ⚠️  PARTIAL - Status: {response.status_code}")
            except Exception as e:
                print(f"  ❌ FAILED - Error: {str(e)}")
    
    # Test requests library
    print(f"\n📡 Testing with requests library")
    print("-" * 40)
    
    session = requests.Session()
    session.verify = certifi.where()
    
    for endpoint in test_endpoints:
        try:
            print(f"  Testing: {endpoint}")
            response = session.get(endpoint, timeout=30)
            if response.status_code == 200:
                print(f"  ✅ SUCCESS - Status: {response.status_code}")
            else:
                print(f"  ⚠️  PARTIAL - Status: {response.status_code}")
        except Exception as e:
            print(f"  ❌ FAILED - Error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🏁 SSL connection test completed!")

if __name__ == "__main__":
    asyncio.run(test_ssl_connections())
