import os
import requests
import argparse

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def main(args):
    response = query({
        "messages": [
            {
                "role": "user",
                "content": "What is the capital of India?"
            }
        ],
        "model": args.model_name # e.g., "deepseek-ai/DeepSeek-V3:together"
    })

    print(response["choices"][0]["message"]['content'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, help="Model name")
    args = parser.parse_args()
    main(args)