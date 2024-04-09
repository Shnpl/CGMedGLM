    # {
    #     "instruction": "现在你是一个医生，请根据患者的问题给出建议：",
    #     "question": "想知道阳痿早泄龟头敏感怎么办？？我性生活时间短，坚持不到的5分钟就射了，要怎么治疗啊，费用会高吗？哪看早泄医院好？这个病是什么原因引起的？可有什么疗能彻底根治？费用会不会很高？阳痿早泄龟头敏感怎么办？",
    #     "answer": "1：对于轻微早泄通过平时的调理，一般是可以恢复的，但是对于中度和严重早泄则一般是要采取综合评价治疗方式。要适当练习相关性生理方面的技术，以彻底消除精神紧张和恐惧心理。2：最好去当地正规的公立医院进行咨询清楚后，实施治疗。男性朋友在日常生活和工作中要注意重新调整自己的情绪，彻底消除由于担心女性会怀孕，或者是担心自己的生殖器太小、性能力还不够等所引起的紧张、自卑与恐惧的心理。男性频繁性交容易导致前列腺炎和尿路感染和导致精神状态不好乏力性欲低下影响工作和练习甚至影响受孕。"
    # },    
    
    
    # {
    #     "conversations": [
    #         {
    #             "role": "system",
    #             "content": "现在你是一个呼吸科医生，请根据患者的问题给出建议："
    #         },
    #         {
    #             "role": "user",
    #             "content": "前几天我的宝宝因咳嗽引起肺炎，咳嗽期间我们给她吃了药可是不见好转，就到医院检查了下才知道得了肺炎。请问肺炎容易治好吗"
    #         },
    #         {
    #             "role": "assistant",
    #             "content": "肺炎是肺部最常见的良性炎症，多继发于上呼吸道感染，多由于病毒、支原体或细菌渗入人体引来的发烫、咳、咳痰及呼吸困难等，肺炎一般经过积极治疗是可以全愈的，您的宝宝的情况已经明确确诊，所以您需要有积极实施治疗。建议您积极实施抗炎及抗病毒治疗，期间留意休息及防寒，多饮温开水，始终保持宝宝的呼吸道通畅，随天气变动多寡衣物，防止受寒，守护呼吸道，防止烟雾、粉尘及有害气体对呼吸道的危害，不定期复检，期望我的解释带给您帮助！"
    #         }
    #     ]
    # },
import json
path_list = [
    "datasets/medical_conversation/MixMedGLM_v0.2/instruction_like/train.json",
    "datasets/medical_conversation/MixMedGLM_v0.2/instruction_like/dev.json",
    "datasets/medical_conversation/MixMedGLM_v0.2/instruction_like/test.json"
]

for path in path_list:
    data_new = []
    with open(path, 'r') as f:
        data = json.load(f)
        for item in data:
            new_obj = {
                "conversations": 
                    [
                        {
                        "role": "system",
                        "content": item['instruction']
                        },
                        {
                            "role": "user",
                            "content": item['question']
                        },
                        {
                            "role": "assistant",
                            "content": item['answer']
                        }
                    ]
            }
            data_new.append(new_obj)
    with open(path.replace('instruction_like', 'dialogue_like'), 'w') as f:
        # Write jsonl
        for item in data_new:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
