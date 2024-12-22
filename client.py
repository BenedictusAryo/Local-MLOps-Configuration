"""
Sample client to test the API endpoint of the model service
"""

import requests

request_body = {
    "Fresh": 12669,
    "Milk": 9656,
    "Grocery": 7561,
    "Frozen": 214,
    "Detergents_Paper": 2674,
    "Delicassen": 1338,
    "Channel": 1,
}
response = requests.post("http://127.0.0.1:8000/predict", json=request_body)
print(f"Status: {response.status_code}\nResponse:\n {response.json()}")
