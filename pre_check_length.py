import json
with open('datasets/medical_conversation/Pinecone628中文医学问答数据集/dev.json','r') as f:
    data = json.load(f)
    print(len(data))