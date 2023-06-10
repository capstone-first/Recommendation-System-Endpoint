import requests
import json

url = "http://127.0.0.1:8000/recommend"

input_data_for_request = {
    "sentence": ["Fruit and Seed Bars"]
}

input_data = json.dumps(input_data_for_request)
response = requests.post(url, data=input_data)

print(response.text)