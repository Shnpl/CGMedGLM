import json
import random
import re

s = "现在你是一个[...]科医生"
pattern = r"现在你是一个\w+科医生"
replacement = "现在你是一个医生"

src_file_list = ["datasets/medical_conversation/MixMedGLM_v0.2/CMD_Andriatria_4k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/CMD_IM_16k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/CMD_OAGD_8k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/CMD_Oncology_4k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/CMD_Pediatric_4k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/CMD_Surgical_8k.json",
                 "datasets/medical_conversation/MixMedGLM_v0.2/Huatuo26m-Lite_42k.json"]

split_ratio = (0.8, 0.1, 0.1)
split_names = ["train", "dev", "test"]
train_data = []
dev_data = []
test_data = []
for path in src_file_list:
    with open(path, 'r') as f:
        data = json.load(f)
        for item in data:
            item['instruction'] = re.sub(pattern, replacement, item['instruction'])
            
        random.shuffle(data)
        L = len(data)
        train_idxs = range(int(L * split_ratio[0]))
        dev_idxs = range(int(L * split_ratio[0]), int(L * (split_ratio[0] + split_ratio[1])))
        test_idxs = range(int(L * (split_ratio[0] + split_ratio[1])), L)
        train_data.extend([data[i] for i in train_idxs])
        dev_data.extend([data[i] for i in dev_idxs])
        test_data.extend([data[i] for i in test_idxs])
with open('train.json', 'w') as f:
    json.dump(train_data, f, indent=4,ensure_ascii=False)
with open('dev.json', 'w') as f:
    json.dump(dev_data, f, indent=4,ensure_ascii=False)
with open('test.json', 'w') as f:
    json.dump(test_data, f, indent=4,ensure_ascii=False)