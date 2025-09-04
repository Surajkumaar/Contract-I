import requests
import time
import os

# Test uploading a contract to verify the API fix
def test_contract_upload():
    # Use an existing PDF file from the uploads directory
    pdf_path = r"C:\Users\91735\Desktop\rag\contract-intelligence\backend\uploads\sample_contract.pdf"
    
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Test file not found: {pdf_path}")
        # Let's try to find any PDF file in the uploads directory
        uploads_dir = r"C:\Users\91735\Desktop\rag\contract-intelligence\backend\uploads"
        if os.path.exists(uploads_dir):
            pdf_files = [f for f in os.listdir(uploads_dir) if f.endswith('.pdf')]
            if pdf_files:
                pdf_path = os.path.join(uploads_dir, pdf_files[0])
                print(f"Using alternate file: {pdf_path}")
            else:
                print("❌ No PDF files found in uploads directory")
                return
        else:
            print("❌ Uploads directory not found")
            return
    
    try:
        # Upload the contract
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            response = requests.post('http://localhost:8000/contracts/upload', files=files)
        
        if response.status_code == 200:
            contract_id = response.json()['contract_id']
            print(f"✅ Contract uploaded successfully! ID: {contract_id}")
            
            # Check the status
            for i in range(60):  # Wait up to 60 seconds
                status_response = requests.get(f'http://localhost:8000/contracts/{contract_id}/status')
                status_data = status_response.json()
                print(f"Status check {i+1}: {status_data}")
                
                if status_data['status'] in ['completed', 'error']:
                    break
                time.sleep(2)
            
            # Get the final result
            if status_data['status'] == 'completed':
                print("✅ Contract processing completed successfully!")
                data_response = requests.get(f'http://localhost:8000/contracts/{contract_id}')
                print(f"Extracted data: {data_response.json()}")
                return True
            elif status_data['status'] == 'error':
                print(f"❌ Error: {status_data.get('error', 'Unknown error')}")
                return False
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        return False

if __name__ == "__main__":
    success = test_contract_upload()
    if success:
        print("\n🎉 API fix successful! OpenRouter connection is working.")
    else:
        print("\n❌ API fix needs more work.")
