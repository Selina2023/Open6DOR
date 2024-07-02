import base64
import requests

# OpenAI API Key
# api_key = "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"
api_key = "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"
api_key = "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"
api_key = "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"
# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def infer_path(prompt, path):
    # Getting the base64 string
    base64_image = encode_image(path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4-vision-preview",
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
    path = "imgs/bana_cup_gsam_cup.jpg"
    response = infer_path(prompt, path)
    print(response.json())
# prompt_path = "pure_prompt.txt"
# import os
# os.makedirs("GPT4V-pure", exist_ok=True)
# import glob, json, os
# paths = glob.glob("result/*.png")
# prompt_ori = open(prompt_path, "r").read()
# total = len(paths)
# for i, path in enumerate(paths):
#     name = path.split("/")[-1].split(".")[0]
#     print(name, i , total)
#     save_path = f"GPT4V-pure/{name}_pure.json"
#     if os.path.exists(save_path):
#         continue
#     # prompt = prompt_ori + open(f"pure_GAPartNet/{name}_pure_GAPartNet.txt", "r").read()
#     prompt = prompt_ori
#     response = infer_path(prompt, path)
#     json.dump(response.json(), open(save_path, "w"))
#     # import pdb; pdb.set_trace()