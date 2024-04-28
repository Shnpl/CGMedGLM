import sys
import dotenv
dotenv.load_dotenv()
sys.path.append(".")
import argparse
import openai
import os
import numpy as np
import pandas as pd
import time
from .crop import crop
# Debugpy
#import debugpy; debugpy.connect(('localhost', 5678))
#import logging; logging.basicConfig(level=logging.DEBUG)

from langchain_openai import ChatOpenAI
openai.base_url = 'http://127.0.0.1:8000/v1/'
openai.api_key ='...'
openai.api_version = "2023-05-15"

use_chain = True
from modules.chain_module import MainChain
chain = MainChain(verbose=False)



choices = ["A", "B", "C", "D"]
from tqdm import tqdm

def softmax(x):
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    return softmax

def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    # prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    prompt = ""
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def eval(args, subject, engine, dev_df, test_df):
    cors = []
    all_probs = []
    answers = choices[:test_df.shape[1]-2]

    for i in tqdm(range(test_df.shape[0])):
        # get prompt and make sure it fits
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end

        while crop(prompt) != prompt:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]

        while True:
            # if use_chain:
            c = engine.invoke(prompt)
            out_ans = c[0]
            break
            # else:
            # #try:
            #     message=[
            #        {
            #            "role": "system",
            #            "content": "You are a helpful assistant."
            #         },
            #         {
            #             "role": "user",
            #             "content": prompt
            #         }
            #     ]
            #     c = engine.invoke(prompt)
            #     c = openai.chat.completions.create(
            #         model='davinci',
            #         messages=message,
            #         max_tokens=10,
            #         logprobs=True,
            #         temperature=0,
            #     )
                
            #     out_ans = c.choices[0].message.content
            
            # except:
            #     print("pausing")
            #     time.sleep(1)
            #     continue

        lprobs = []
        
        for ans in answers:
            # try:
            #     #lprobs.append(c.choices[0].logprobs.top_logprobs[-1]["{}".format(ans)])
                
                
            # except:
            #     print("Warning: {} not found. Artificially adding log prob of -100.".format(ans))
            #     lprobs.append(-100)
            if ans == out_ans:
                lprobs.append(0)
            else:
                lprobs.append(-100)
        pred = {0: "A", 1: "B", 2: "C", 3: "D"}[np.argmax(lprobs)]
        probs = softmax(np.array(lprobs))

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)
        
        # if i > 10:
        #     break

    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs

def main(args,engine):
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    engine_name = engine.__class__.__name__
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(engine_name))):
        os.mkdir(os.path.join(args.save_dir, "results_{}".format(engine_name)))

    print(subjects)
    print(args)

    all_cors = []

    for subject in subjects:
        dev_df = pd.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.ntrain]
        test_df = pd.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)

        cors, acc, probs = eval(args, subject, engine, dev_df, test_df)
        all_cors.append(cors)

        test_df["{}_correct".format(engine)] = cors
        for j in range(probs.shape[1]):
            choice = choices[j]
            test_df["{}_choice{}_probs".format(engine, choice)] = probs[:, j]
        test_df.to_csv(os.path.join(args.save_dir, "results_{}".format(engine_name), "{}.csv".format(subject)), index=None)
        break

    weighted_acc = np.mean(np.concatenate(all_cors))
    print("Average accuracy: {:.3f}".format(weighted_acc))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--data_dir", "-d", type=str, default="data")
    parser.add_argument("--save_dir", "-s", type=str, default="results")
    args = parser.parse_args()
    main(args)

