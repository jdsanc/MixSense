import os
import requests
import json

# Get token from environment
token = os.environ.get("HF_TOKEN")
if not token:
    raise ValueError("HF_TOKEN environment variable not set")

# API URL for the model
model_id = "meta-llama/Llama-2-70b-chat-hf"
url = f"https://api-inference.huggingface.co/endpoints/meta-llama/Llama-2-70b-chat-hf/invoke"  # Using endpoints API

def llm(query):
    # Parameters for generation
    parameters = {
        "max_new_tokens": 5000,
        "temperature": 0.01,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False
    }
    
    # Format prompt using Llama 2's chat format
    prompt = f"""<s>[INST] <<SYS>>
You are a helpful and smart assistant. You accurately provide answer to the provided user query.
<</SYS>>

Here is the query: ```{query}```.
Provide precise and concise answer.[/INST]"""
    
    headers = {
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    }
    
    # Prepare payload
    payload = {
        "inputs": prompt,
        "parameters": parameters
    }
    
    print(f"\nSending request to: {url}")
    print(f"Headers: {headers}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    try:
        # Make the API call
        response = requests.post(url, headers=headers, json=payload)
        print(f"\nResponse Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Content: {response.text}")
        
        # Check status code first
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.text}"
        
        # Try to parse JSON response
        try:
            response_json = response.json()
            print(f"\nParsed JSON response: {json.dumps(response_json, indent=2)}")
            
            if isinstance(response_json, list) and len(response_json) > 0:
                generated_text = response_json[0].get('generated_text', '')
                if generated_text:
                    return generated_text.strip()
                else:
                    return "Error: No generated text in response"
            else:
                return f"Error: Unexpected response format - {response_json}"
                
        except json.JSONDecodeError as e:
            print(f"\nFailed to parse JSON: {str(e)}")
            return f"Error: Invalid JSON response - {str(e)}"
            
    except requests.exceptions.RequestException as e:
        print(f"\nRequest failed: {str(e)}")
        return f"Error: Request failed - {str(e)}"


if __name__ == "__main__":
    # Test the API
    query = "What is the capital of India?"
    print("\nTesting with query:", query)
    result = llm(query)
    print("\nFinal result:", result)