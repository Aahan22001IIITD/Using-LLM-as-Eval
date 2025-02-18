import os
import requests

# Set Ollama server host
os.environ["OLLAMA_HOST"] = "192.168.23.138:11439"
OLLAMA_HOST = os.environ["OLLAMA_HOST"]

# API endpoint to list models
url = f"http://{OLLAMA_HOST}/api/tags"

try:
    response = requests.get(url)
    if response.status_code == 200:
        models = response.json().get("models", [])
        if models:
            print("Available Models:")
            for model in models:
                print(f"- {model['name']}")
        else:
            print("No models found.")
    else:
        print(f"Failed to fetch models. Status Code: {response.status_code}, Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
