import dotenv
dotenv.load_dotenv()
import os
import openai

import json
from openai import OpenAI
from tqdm import tqdm
import logging
import requests



import requests
import json

def query_openai(messages, use_proxy=True):
    url = "https://api.openai.com/v1/chat/completions"
    api_key = os.environ['OPENAI_API_KEY']
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    data = {
        "model": "gpt-4",
        "messages": messages,
        "max_tokens": 512,
        "temperature":1,
        "top_p":1.0,
        "frequency_penalty":0.0,
        "presence_penalty":0.0,
    }

    proxies = {
        'http': 'http://192.168.2.2:10809',
        'https': 'http://192.168.2.2:10809',
    } if use_proxy else None

    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    return response.json()['choices'][0]['message']['content'].replace("\n","")

# def get_answer():
#     '''
#     gpt-4
#     '''
#     if not os.environ['http_proxy'] or not os.environ['https_proxy']:
#         raise 'Please set http_proxy and https_proxy environment variables'

def get_answer(question:str):
    '''
    chatglm3
    '''
    openai.base_url= "http://127.0.0.1:8000/v1/"
    diagnosis_prompt = "示例：病人：吸入性肺炎，患者男80岁肺部严重吸入感染住院10余天上呼吸机抗生素治疗每日吸痰效果不佳体质极弱体四肢浮肿且排尿不畅褥疮严重求助谢谢。医生：您好！老人目前的营养状况极差，看症状，恐怕有多器官衰竭，除了抗感染和支持治疗外，没有什么更好的办法目前。\n病人：不孕不育，做完子宫输卵管造影后造影结果是子宫腔形态正常，内膜光整，右侧输卵管上举，双侧输卵管伞端部分粘连，内有造影剂残留，盆腔弥散局限。提示：双侧输卵管通而不畅。医生：如果输卵管不通畅的话，人工受精的受孕几率也比较低，建议行中药输卵管治疗2-3个月，如果仍然不孕，可以行试管婴儿助孕治疗。\n病人：皮肤病，全身脸部5岁开始褐色班越来越多20多岁开始长像块一样的有些红不疼不痒无不适症状请仔细看我的图片拍了5张。医生：神经纤维瘤可能性较大，具体还要配合其他检查，另外注意孩子有无眼部的异常情况，此病常有伴发。也可能没有。治疗起来暂无特效治愈办法，手术切除减瘤为主。\n病人：不会走路，孩子可能早产，催产素生下，眼睛没张开。三天后下奶我乳腺增生，医生建议我母乳喂养，也许是奶水的缘故孩子一直拉肚子到六十天，然后住院，确定拉肚子引起支气管肺炎。孩子体能方面发育比正常小孩慢，十六个月才会爬的。智能方面感觉超越同龄人，但语言方面欠缺。现在在别人的帮助下能走十几米，走起来脚后跟不着地，可是不敢独自走路、站立、下蹲，每人牵就跪着。医生：周一到周五下午3点以后可以来北京清华大学玉泉医院神经外科三病区医生办公室找我。孩子的情况可能需要手术。\n病人：剖腹产刀疤，患者女，33岁，身体状况良好。病情09年6月剖腹产手术，刀疤凸起，宽约1厘米，长约5厘米。医生：可以治疗，如果不是疤痕体质，术后效果较好。可以到我们医院治疗。最佳治疗方案，要我看过之后，方能给出。费用2000到5000元不等。\n现在你是一个富有经验的医生，一个医疗专业人士，急需你根据问题给出决断性的回答："
    messages=[
                {"role": "system", "content": diagnosis_prompt},
                {"role": "user", "content":question},
            ]
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
    return ans

def judge_answer(question:str,answer:str,fewshot:bool=True):
    '''
    if hallucination exists:True
    else: False
    '''
    
    
    with open('utils/med_hall_judge/data/hallucination_judge_prompt.txt','r') as f:
        prompt = f.read()
    if fewshot:
        with open('utils/med_hall_judge/data/hallucination_judge_prompt_fewshot_extra.txt','r') as f:
            prompt += f.read()
    content = "病人："+question+"医生："+answer
    messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": content},
        ]
    ans = query_openai(messages, use_proxy=True)
    
    if "NNN" in ans:
        return False
    elif "YYY" in ans:
        return True
    else:
        return None
    
def main():
    fewshot= False
    with open('trunk_0.json','r') as f:
        src_data = json.load(f)
    all_num = 0
    right_num = 0
    wrong_num = 0
    ub_num = 0
    hard_questions = []
    src_data = src_data
    for item in tqdm(src_data[:]):
        question = item[0]
        try:
            ans = get_answer(question)
            res = judge_answer(question,ans,fewshot=fewshot)
        except:
            with open('remain.json','w') as f:
                json.dump(src_data,f,ensure_ascii=False,indent=4)
            if not os.path.exists('hard_questions.json'):
                with open('hard_questions.json','w') as f:
                    json.dump(hard_questions,f,ensure_ascii=False,indent=4)
            else:
                with open('hard_questions.json','r') as f:
                    data = json.load(f)
                    data += hard_questions
                with open('hard_questions.json','w') as f:
                    json.dump(data,f,ensure_ascii=False,indent=4)
        if res == False:
            right_num += 1
        elif res == True:
            wrong_num += 1
            hard_questions.append({
                "Question":item[0],
                "Original":item[1],
                "Generated":ans
            })
        else:
            ub_num += 1
        src_data.remove(item)
    with open('remain.json','w') as f:
        json.dump(src_data,f,ensure_ascii=False,indent=4)
    if not os.path.exists('hard_questions.json'):
        with open('hard_questions.json','w') as f:
            json.dump(hard_questions,f,ensure_ascii=False,indent=4)
    else:
        with open('hard_questions.json','r') as f:
            data = json.load(f)
            data += hard_questions
        with open('hard_questions.json','w') as f:
            json.dump(data,f,ensure_ascii=False,indent=4)
            
        
    print(f"Total: {all_num}, Right: {right_num}, Wrong: {wrong_num}, UB: {ub_num}")

if __name__ == "__main__":
    main()