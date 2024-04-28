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
        "temperature":0,
        "top_p":1.0,
        "frequency_penalty":0.0,
        "presence_penalty":0.0,
    }

    proxies = {
        'http': 'http://192.168.2.2:10809',
        'https': 'http://192.168.2.2:10809',
    } if use_proxy else None

    response = requests.post(url, headers=headers, json=data, proxies=proxies)
    try:
        ans = response.json()['choices'][0]['message']['content'].replace("\n","")
    except:
        print(response.json())
        raise Exception("OpenAI API failed")
    return ans