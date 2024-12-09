# Assuming you have access to the Gemini API
from google.cloud import aiplatform

# Initialize the API client
aiplatform.init(project='genai-prep')

# Define your request
request = {
    "prompt": "Write a poem about a cat.",
    "model": "gemini-1.5-pro-001"
}

# Send the request to the Gemini API
response = aiplatform.call_model(model_name='gemini-1.5-pro-001', request=request)

# Process the response
print(response.text)
