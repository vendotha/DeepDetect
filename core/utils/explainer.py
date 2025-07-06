import requests

API_KEY = "sk-b8249cc7e1ca4a36a7392c25b78baa77"
API_URL = "https://api.deepseek.com/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}


def get_explanation(prediction: str):
    prompt = f"Explain in simple terms why a video or image might be classified as '{prediction}' by a deepfake detector model."

    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Could not fetch explanation: {str(e)}"
