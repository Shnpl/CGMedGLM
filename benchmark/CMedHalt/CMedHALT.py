import sys
sys.path.append('.')
from modules.chain_module import MainChain,SimpleChain
import pandas as pd
from benchmark.CMedHalt.judge_hallucination_qa import GPT4Judge
path = 'benchmark/CMedHalt/data/qa/NewHallu_T.xlsx'
q_df = pd.read_excel(path)
chain = SimpleChain()
#chain = MainChain()
judge = GPT4Judge()
hallu_num = 0
for i in range(0,len(q_df)):
    c = {
        "A":"",
        "B":"",
        "C":"",
        "D":"",
        "E":""
    }
    q_template,right_index,wrong_index,c["A"],c["B"],c["C"],c["D"],c["E"]=q_df.iloc[i]
    all_others = "<"
    for key in c:
        if c[key] == "nan":
            continue
        else:
            all_others += str(c[key])+"，"
    all_others = all_others[:-1]+">"
    question = q_template.format(all_others,"哪个")        
    #question = q_template.format(all_others,c[wrong_index])
    ref_ans = c[right_index]
    ans = chain.invoke(question)
    with open('output.txt','a') as f:
        f.write(f"Question: {question},\n Ref Answer: {ref_ans}\n, Answer: {ans}\n,current_total_num:{i+1}\n")
    #judge_result = judge.invoke(question,c[right_index],ans)
    # if 'YYY' in judge_result:
    #     hallu_num += 1
    # print(f"Question: {question},\n Ref Answer: {ref_ans}\n, Answer: {ans}\n, Judge Result: {judge_result}\n,current_total_num:{i+1},current_hallu_num:{hallu_num}")