import json
import os
from tqdm import tqdm
import numpy as np
import torch
import json
import pickle
import random
from sklearn.cluster import KMeans
from utils.clustering import DisjointSet
from hashlib import md5

def cmd_load_data(file_path:str):
    with open(file_path,'r',encoding='utf-8') as f:
        data = json.load(f)
    return data
def cmd_question_concat(item):
    return item['department'] + " " + item['title'] + " " + item['question']
def cmd_rephrase(item):
    new_item = {
            "instruction":f"现在你是一个{item['department']}科医生，请根据患者的问题给出建议：".replace("科科","科"),
            "question":item['question'],
            "answer":item['answers']
        }
    return new_item

def huatuo_lite_load_data(file_path:str):
    """
    jsonl
    """
    data = []
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            data.append(item)
    return data
def huatuo_lite_question_concat(item):
    return item['label'] + " " + item['question']+ " "+ item["related_diseases"]
def huatuo_lite_rephrase(item):
    new_item = {
            "instruction":f"现在你是一个{item['label']}科医生，请根据患者的问题给出建议：".replace("科科","科"),
            "question":item['question'],
            "answer":item['answer']
        }
    return new_item

def main(original_path:str,sim_threshold:float=0.85,dataset_name = 'cmd',target_len=None):
    question_concat = {
        'cmd':cmd_question_concat,
        'huatuo_lite':huatuo_lite_question_concat
    }
    load_data = {
        'cmd':cmd_load_data,
        'huatuo_lite':huatuo_lite_load_data
    }
    rephrase={
        'cmd':cmd_rephrase,
        'huatuo_lite':huatuo_lite_rephrase
    }
    data = load_data[dataset_name](original_path)
    
    
    cache_name = md5(original_path.encode()).hexdigest()
    cache_path = os.path.join('cache',cache_name)
    print(f"Cache name: {cache_name}")
    if os.path.exists(cache_path):
        with open(cache_path,'rb') as f:
            new_q_vec_list = pickle.load(f)
        print("Loaded from cache")
    else:
        print("No cache found")
        from sentence_transformers import SentenceTransformer, util
        model = SentenceTransformer('distiluse-base-multilingual-cased')


        print("Original Length: ",len(data))
        new_q_list = []
        new_q_vec_list = []
        for item in tqdm(data):
            text = question_concat[dataset_name](item)
            new_q_list.append(text)
            q_embedding = model.encode(text)
            new_q_vec_list.append(q_embedding)
        with open(cache_path,'wb') as f:
            pickle.dump(new_q_vec_list,f)
    if not os.path.exists(f'out/{cache_name}'):
        os.makedirs(f'out/{cache_name}')
        from sentence_transformers import util
        cos_sim_matrix = util.pytorch_cos_sim(new_q_vec_list, new_q_vec_list).numpy()
        B = len(cos_sim_matrix)
        # 应用阈值
        cos_sim_matrix = cos_sim_matrix > sim_threshold

        # 初始化并查集
        ds = DisjointSet(B)

        # 合并索引
        for i in tqdm(range(B), position=0, leave=False):
            for j in range(i+1, B):
                if cos_sim_matrix[i][j]:
                    ds.union(i, j)

        # 构建块
        blocks = {}
        for i in range(B):
            root = ds.find(i)
            if root not in blocks:
                blocks[root] = []
            blocks[root].append(i)

        # 输出块的数量
        print("Unique Questions: ", len(blocks))

        for idx,block in enumerate(blocks):
            with open(f'out/{cache_name}/tmp_unique_{idx}.json', 'w') as f:
                json.dump([data[i] for i in blocks[block]], f, ensure_ascii=False, indent=4)
    else:
        print("Output directory already exists")
    blocks = []
    for file in os.listdir(f'out/{cache_name}'):
        with open(os.path.join(f'out/{cache_name}',file),'r') as f:
            block = json.load(f)
            blocks.append(block)
    new_data = []
    for block in blocks:
        new_item =rephrase[dataset_name](block[0])
        new_data.append(new_item)
    if target_len:
        idxs = random.choices(range(len(new_data)),k=target_len)
        new_data = [new_data[i] for i in idxs]
    with open(f'out/{cache_name}.json','w') as f:
        json.dump(new_data,f,ensure_ascii=False,indent=4)
    
if __name__ == '__main__':
    original_file_path = 'datasets/medical_conversation/Huatuo-26M/Huatuo26M-Lite/format_data.jsonl'
    main(original_path=original_file_path,
                         dataset_name='huatuo_lite',target_len=42000)
    