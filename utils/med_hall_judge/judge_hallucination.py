import dotenv
dotenv.load_dotenv()
import openai
import os
import json
from openai import OpenAI
from tqdm import tqdm
import logging
import requests

    

model = 'chatglm3'
fewshot= False
num_samples = 50 # 2*50


if model == 'gpt-4':
    if not os.environ['http_proxy'] or not os.environ['https_proxy']:
        exit('Please set http_proxy and https_proxy environment variables')
if model == 'chatglm3':
    
    # local model
    url = "http://127.0.0.1:5000/v1/chat/completions"

    headers = {
        "Content-Type": "application/json"
    }

with open('utils/med_hall_judge/data/hallucination_judge_prompt.txt','r') as f:
    prompt = f.read()
if fewshot:
    with open('utils/med_hall_judge/data/hallucination_judge_prompt_fewshot_extra.txt','r') as f:
        prompt += f.read()
with open('datasets/GPT-Judge/gen_output.json','r') as f:
    src_data = json.load(f)
all_num = 0
right_num = 0
wrong_num = 0
right_as_wrong_num = 0
wrong_as_right_num = 0


src_data = src_data[:num_samples]
for item in tqdm(src_data):
    for j in ["Original","Generated"]:
        all_num += 1
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": item['Question']+item[j]},
        ]
        if model == 'gpt-4':    
            response = openai.chat.completions.create(
            model="gpt-4", # Choose the appropriate model
            messages=messages,
            temperature=1,
            max_tokens=512,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            )
            ans = response.choices[0].message.content.replace("\n","")
        elif model =='chatglm3':
            data = {
                "mode": "instruct",
                "character": "Example",
                "messages": messages,
                "max_tokens": 512
            }
            response = requests.post(url, headers=headers, json=data, verify=False)
            ans = response.json()['choices'][0]['message']
            ans = ans['content']
            print(messages)
            print(ans)
            print(f"Type:{j}")
            
            
        
        if j == "Original":
            if "NNN" in ans:
                right_num += 1
            else:
                wrong_num += 1
                right_as_wrong_num += 1
        elif j == "Generated":
            if "YYY" in ans:
                right_num += 1
            else:
                wrong_num += 1
                wrong_as_right_num += 1
            
print(f"Total: {all_num}, Right: {right_num}, Wrong: {wrong_num}, Right as Wrong: {right_as_wrong_num}, Wrong as Right: {wrong_as_right_num}")