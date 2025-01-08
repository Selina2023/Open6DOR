import base64
import requests

# OpenAI API Key
import os
API_KEY = os.getenv("API_KEY")
if API_KEY is None:
    raise ValueError("please set API_KEY environment variable by running `export API_KEY=XXXX`")
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def infer_path(prompt, path):
    # Getting the base64 string
    base64_image = encode_image(path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # print(response.json())
    return response


if __name__ == "__main__":
    prompt = "descripbe this image"
    path = "./vision/1.jpg"
    response = infer_path(prompt, path)
    print(response.json())
