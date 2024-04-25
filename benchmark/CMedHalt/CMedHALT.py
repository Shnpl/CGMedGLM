from modules.chain_module import MainChain,SimpleChain
import pandas as pd
from benchmark.CMedHalt.judge_hallucination_qa import GPT4Judge
q_df = pd.read_csv('utils/med_hall_judge/data/NewHallu.CSV')
chain = MainChain()
judge = GPT4Judge()
hallu_num = 0
for i in range(len(q_df)):
    c = [0]*4
    simple_q,q_template,c[0],c[1],c[2],c[3],right_index,wrong_index=q_df.iloc[i]
    right_index = int(right_index)
    wrong_index = int(wrong_index)
    question = q_template.format(c[wrong_index])
    ref_ans = c[right_index]
    ans = chain.invoke(question)
    judge_result = judge.invoke(question,c[right_index],ans)
    if 'YYY' in judge_result:
        hallu_num += 1
    print(f"Question: {question}, Ref Answer: {ref_ans}, Answer: {ans}, Judge Result: {judge_result},current_total_num:{i+1},current_hallu_num:{hallu_num}")