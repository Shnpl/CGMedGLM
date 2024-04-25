import dotenv
dotenv.load_dotenv()
import openai
import os
import json
from tqdm import tqdm
if not os.environ['http_proxy'] or not os.environ['https_proxy']:
    exit('Please set http_proxy and https_proxy environment variables')


with open('utils/med_hall_judge/data/hallucination_gen_prompt.txt','r') as f:
    prompt = f.read()
with open('utils/med_hall_judge/data/gen_src.json','r') as f:
    src_data = json.load(f)
gen_data = []
for item in tqdm(src_data):
    prompt_item = prompt+item[0]+item[1]
    print(prompt_item)
    messages=[
        {"role": "system", "content": prompt_item},
    ]
    response = openai.chat.completions.create(
    model="gpt-4", # Choose the appropriate model
    messages=messages,
    temperature=0.7,
    max_tokens=512,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    gen_data.append(
        {
            "Question":item[0].replace("\n",""),
            "Original":item[1].replace("\n",""),
            "Generated":response.choices[0].message.content.replace("\n","")
        }
    )
with open('utils/med_hall_judge/data/gen_output.json','a') as f:
    json.dump(gen_data,f)


