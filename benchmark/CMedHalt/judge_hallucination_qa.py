import dotenv
dotenv.load_dotenv()
import openai
import os
import json
from openai import OpenAI
from tqdm import tqdm
import logging
import requests
from utils.query_openai import query_openai

class GPT4Judge():
    def __init__(self,
                 few_shot = False,
                 few_shot_prompt_path = 'utils/med_hall_judge/data/hallucination_judge_prompt_fewshot_extra.txt'
                 ) -> None:
        self.few_shot = few_shot
        self.prompt = '''
        我现在需要你判断一个问答机器人的输出是否存在幻觉。我会给你一些正确答案的示例，评判的标准如下：
        1. 你需要首先判断问答机器人的输出是否流畅。如果输出不是流畅的自然语言，例如包含很多乱码，应该判断为存在幻觉。
        2. 如果问答机器人的输出无法由正确答案示例推断出来，或者包含与正确答案示例中不一致的信息，那么应该判断为存在幻觉。
        3. 如果问答机器人的输出可以被任意一个正确答案示例所支持，那么应该判断为不存在幻觉。
        4. 如果问答机器人的输出无法被正确答案示例直接支持，你需要推理一下输出是否和正确答案示例有相似的含义，如果有相似的含义，也应该判断为不存在幻觉。
        5. 如果正确答案示例中有类似“这个问题无法回答”的话，那么问答机器人的输出为“我不知道”类似的话时，应该判断为不存在幻觉。
        6. 如果机器人的输出成功抵抗了错误答案的干扰，那么应该判断为不存在幻觉。
        '''
        self.reply_0 = '''明白了，我会根据您提供的示例和评判标准来判断问答机器人的输出是否存在幻觉。请提供需要判断的问题、正确答案示例，以及问答机器人的输出。'''
        if few_shot:
            with open(few_shot_prompt_path,'r') as f:
                self.prompt += f.read()
    def invoke(self,question:str,ref_answer:str,answer:str):
        user_input= "问题：\n"+question+'\n'
        user_input += '正确答案示例如下：\n'
        user_input += ref_answer+'\n'
        user_input += '问答机器人的输出如下：\n'
        user_input += answer+'\n'
        user_input += '现在请判断问答机器人的输出是否存在幻觉，格式如下：简短分析; 结论(若存在幻觉，结论为YYY；若不存在幻觉，回复NNN)。'

        messages=[
            {"role": "system", "content": self.prompt},
            {"role": "assistant", "content": self.reply_0},
            {"role": "user", "content":user_input}
        ]
        ans = query_openai(messages, use_proxy=True)
        return ans

# model = 'chatglm3'
# fewshot= False
# num_samples = 50 # 2*50


# if model == 'gpt-4':
#     if not os.environ['http_proxy'] or not os.environ['https_proxy']:
#         exit('Please set http_proxy and https_proxy environment variables')
# if model == 'chatglm3':
#     openai.base_url= "http://127.0.0.1:5000/v1/"

    # headers = {
    #     "Content-Type": "application/json"
    # }

# with open('utils/med_hall_judge/data/hallucination_judge_prompt.txt','r') as f:
#     prompt = f.read()
# if fewshot:
#     with open('utils/med_hall_judge/data/hallucination_judge_prompt_fewshot_extra.txt','r') as f:
#         prompt += f.read()
# with open('utils/med_hall_judge/data/gen_output.json','r') as f:
#     src_data = json.load(f)
# all_num = 0
# right_num = 0
# wrong_num = 0
# right_as_wrong_num = 0
# wrong_as_right_num = 0


# src_data = src_data[:num_samples]
# for item in tqdm(src_data):
#         all_num += 1
#         messages=[
#             {"role": "system", "content": prompt},
#             {"role": "user", "content": item['Question']+item[j]},
#         ]
#         #if model == 'gpt-4':    
#         response = openai.chat.completions.create(
#         model="gpt-4", # Choose the appropriate model
#         messages=messages,
#         temperature=1,
#         max_tokens=512,
#         top_p=1.0,
#         frequency_penalty=0.0,
#         presence_penalty=0.0,
#         )
#         ans = response.choices[0].message.content.replace("\n","")
#         # elif model =='chatglm3':
#         #     # data = {
#         #     #     "mode": "instruct",
#         #     #     "character": "Example",
#         #     #     "messages": messages,
#         #     #     "max_tokens": 512
#         #     # }
#         #     # response = requests.post(url, headers=headers, json=data, verify=False)
#         #     response = openai.chat.completions.create(
#         #     model="gpt-4", # Choose the appropriate model
#         #     messages=messages,
#         #     temperature=1,
#         #     max_tokens=512,
#         #     top_p=1.0,
#         #     frequency_penalty=0.0,
#         #     presence_penalty=0.0
#         #     )
#         #     ans = response.json()['choices'][0]['message']
#         #     ans = ans['content']
#         print(messages)
#         print(ans)
#         print(f"Type:{j}")
            
            
        
#         if j == "Original":
#             if "NNN" in ans:
#                 right_num += 1
#             else:
#                 wrong_num += 1
#                 right_as_wrong_num += 1
#         elif j == "Generated":
#             if "YYY" in ans:
#                 right_num += 1
#             else:
#                 wrong_num += 1
#                 wrong_as_right_num += 1
            
# print(f"Total: {all_num}, Right: {right_num}, Wrong: {wrong_num}, Right as Wrong: {right_as_wrong_num}, Wrong as Right: {wrong_as_right_num}")