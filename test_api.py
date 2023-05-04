import requests
import json

url = "http://127.0.0.1:5000/predict"

# Replace this with your own test data
data = {
    "Product_Id": "e21406c9-5bb9-4bbe-b2e2-72366ee5d5cd",
    "Net_Weight": 94.708971,
    "Size": "A",
    "Value": 8531.628533,
    "Storage": 0,
    "F1": 58.037719,
    # Add remaining feature values here...
}

response = requests.post(url, json=data)
print(response.status_code)
print(response.json())
