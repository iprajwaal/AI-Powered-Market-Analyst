import requests

# Replace with your actual API key
api_key = " "

# Replace with your actual project ID and region
project_id = "genai-prep"
region = "us-central1"

# Construct the API endpoint URL
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-001:generateContent?key={api_key}"

# Prepare the request data
data = {
    "contents": [
        {
            "role": "user",
            "parts": [
                {
                    "text": "Hello, world!"
                }
            ]
        }
    ]
}

# Send the request
response = requests.post(url, json=data)

# Process the response
print(response.json())
