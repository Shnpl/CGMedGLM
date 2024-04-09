import json
with open('train.json','r') as f:
    data = json.load(f)
    print(len(data))