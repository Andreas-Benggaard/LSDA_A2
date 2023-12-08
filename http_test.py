import requests
import json

url =  "http://localhost:5000/invocations"

test = '{"columns": ["Time"], "data": [["2021-04-15T20:00:00"]]}'

headers = {'Content-Type': 'application/json'}

response = requests.post(url, data=test, headers=headers)

print(response.text)