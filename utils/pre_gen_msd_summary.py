import json
import tiktoken
from tqdm import tqdm
with open("datasets/medical_knowledgebase_content/handbooks_guideline/msd_disease_info.json") as f:

    data = json.load(f)
sum_data = {}
for key in tqdm(data):
    new_dict = {}
    if '概述' in data[key]:
        new_dict['概述'] = data[key]['概述']
    if '症状和体征' in data[key]:
        if '概述' in data[key]['症状和体征']:
            new_dict['症状和体征'] = data[key]['症状和体征']['概述']
        else:
            new_dict['症状和体征'] = data[key]['症状和体征']
    if '诊断' in data[key]:
        if '概述' in data[key]['诊断']:
            new_dict['诊断'] = data[key]['诊断']['概述']
        else:
            new_dict['诊断'] = data[key]['诊断']
        
    
    sum_data[key] = new_dict
with open("tmp.json","w") as f:
        json.dump(sum_data,f,ensure_ascii=False,indent=4)
with open("tmp.json") as f:
    data = f.read()
    enc = tiktoken.encoding_for_model("gpt-4")
    encoding =enc.encode(data)
    print(len(encoding))
