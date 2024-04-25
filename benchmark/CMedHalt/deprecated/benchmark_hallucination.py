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
from utils.query_openai import query_openai


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
    num_samples = 50 # 2*50
    with open('utils/med_hall_judge/data/gen_output.json','r') as f:
        src_data = json.load(f)
    all_num = 0
    right_num = 0
    wrong_num = 0
    ub_num = 0

    src_data = src_data[:num_samples]
    for item in tqdm(src_data):
        question = item["Question"]
        ans = get_answer(question)
        res = judge_answer(question,ans)
        if res == False:
            right_num += 1
        elif res == True:
            wrong_num += 1
        else:
            ub_num += 1
                
    print(f"Total: {all_num}, Right: {right_num}, Wrong: {wrong_num}, UB: {ub_num}")

if __name__ == "__main__":
    main()
    
'根据您提供的信息，您的宝宝出现了一个黑毛痣。一般而言，黑毛痣会在出生后的几个月内形成，通常会在青春期达到高峰期。在您所传递的照片中，我注意到宝宝的痣呈现出深褐色，并拥有黑色的毛发。为了确保诊断的准确性，如果方便的话，建议带宝宝去医院接受进一步体检。关于何时治疗，一般来说，越早治疗，效果越好，因为随着年龄的增长，痣可能有恶化的风险。例如，痣细胞可能在某种程度上具有生长潜力和转变成为恶性癌的风险。因此，在您的医生建议下，及早进行手术或其他适当的治疗措施，可能会减少潜在风险和健康问题。至于最佳的治疗方法，则取决于痣的大小、位置和深度等因素。一般来说，外科手术是最常用的治疗方法，可以通过切除痣细胞以降低恶变风险，同时消除外观上的影响。另外，激光或冷冻疗法也可以用来治疗小而浅的黑毛痣，但疗效可能不如手术。具体的治疗方案需要由制定。最后，感谢您愿意分享宝宝的照片。如有需要，我会尽我所能为您提供帮助和建议。祝您的宝宝健康成长！'