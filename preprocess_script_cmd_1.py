import json
import os
from tqdm import tqdm
import numpy as np
import random
import json
import pickle
from sklearn.cluster import KMeans
dir_path = 'out'
files = os.listdir(dir_path)
files = [os.path.join(dir_path,file) for file in files]

data = []
for file in tqdm(files):
    with open(file,'r') as f:
        item = json.load(f)[0]
        new_item = {
            "instruction":f"现在你是一个{item['department']}科医生，请根据患者的问题给出建议：".replace("科科","科"),
            "question":item['title']+'？'+item['question'],
            "answer":item['answers']
        }
        data.append(new_item)


print("Original Length: ",len(data))
target_len = 16000
idxs = random.choices(range(len(data)),k=target_len)
data = [data[i] for i in idxs]
print("New Length: ",len(data))
with open('CMD_IM_16k.json','w') as f:
    json.dump(data,f,ensure_ascii=False,indent=4)