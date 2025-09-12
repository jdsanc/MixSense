import os
import requests

# Get token from environment
token = os.environ.get("HF_TOKEN")

# API URL for the model
url = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"

def llm(query):
  parameters = {
      "max_new_tokens": 5000,
      "temperature": 0.01,
      "top_k": 50,
      "top_p": 0.95,
      "return_full_text": False
      }
  
  prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful and smart assistant. You accurately provide answer to the provided user query.<|eot_id|><|start_header_id|>user<|end_header_id|> Here is the query: ```{query}```.
      Provide precise and concise answer.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
  
  headers = {
      'Authorization': f'Bearer {token}',
      'Content-Type': 'application/json'
  }
  
  prompt = prompt.replace("{query}", query)
  
  payload = {
      "inputs": prompt,
      "parameters": parameters
  }
  
  response = requests.post(url, headers=headers, json=payload)
  response_text = response.json()[0]['generated_text'].strip()

  return response_text


query = "What is the capital of India?"
print(llm(query))