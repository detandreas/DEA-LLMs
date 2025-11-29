import requests
import json

API_KEY = "aa_iokTTocVLAyHukrcqDZSEVxwhoiRcvsN"

url = "https://artificialanalysis.ai/api/v2/data/llms/models"
headers = {"x-api-key": API_KEY}

response = requests.get(url, headers=headers)

# Έλεγχος επιτυχίας
if response.status_code == 200:
    data = response.json()

    # Pretty save σε αρχείο
    with open("models2.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print("Saved to models2.json")
else:
    print("Error:", response.status_code, response.text)
