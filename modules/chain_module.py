import dotenv
dotenv.load_dotenv()
import os
import json
import pickle
from tqdm import tqdm

from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.api.base import APIChain


class SimpleChain():
    def __init__(self) -> None:
        pass
        self.chatmodel = ChatOpenAI(model="gpt-4",
                            base_url=os.environ['LOCAL_BASE_URL'],
                            api_key=os.environ['LOCAL_API_KEY'],temperature=1,max_tokens=512)
        self.main_component = ChatPromptTemplate.from_messages([
            ("system", "现在你是一个医生，请根据患者的问题给出建议："),
            ("user", "{input}")
        ])
        self.main_chain = self.main_component |self.chatmodel|StrOutputParser()
    def invoke(self,input_data):
        print("===输入===")
        print(input_data)
        print("\n")
        x = self.main_chain.invoke({"input": input_data})
        print("===输出===")
        print(x)
        return x

class MainChain():
    def __init__(self,verbose = True) -> None:
        self.verbose = verbose
        self.chatmodel = ChatOpenAI(model="gpt-4",
                            base_url=os.environ['LOCAL_BASE_URL'],
                            api_key=os.environ['LOCAL_API_KEY'],temperature=1,max_tokens=512).bind(logprobs=True)


        self.summarize_component = ChatPromptTemplate.from_messages([
            ("system", "你是一个医生，请从下列信息中提取出病人的病史。\n格式：\n性别：__。\n年龄：__。\n主诉：__。\n现病史：__。\n既往病史：__。\n若无信息，请填写无。"),
            ("user", "{input}")
        ])    
        # MSD Disease Info Stucture:
        # * "不稳定型心绞痛":
        #     |--"概述":Required
        #     |--"症状和体征":
        #     |     |--"概述"
        #     |--"病因"
        #     |     |--"概述"
        #     |--"诊断"
        #     |     |--"概述"
        #     |--"预后"
        #     |     |--"概述"
        #     |--"治疗"
        #     |     |--"概述"
        #           |--"院前治疗"
        #           |--"住院管理"
        #           |--"不稳定型心绞痛药物治疗"
        #           |--"不稳定型心绞痛再灌注治疗"
        #           |--"康复和出院后治疗"
        self.first_round_diagnosis_component = ChatPromptTemplate.from_messages([
            ("system", "根据描述，给出一个初步的诊断以供资料检索。按重要性降序排列。格式：逗号分隔的疾病名称。若无法诊断，请填写无。"),
            ("user", "{input}")
        ])

        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=os.path.join('models','embedding_models','BAAI--bge-large-zh-v1.5'))

        if os.path.exists("cache/msd_disease_info_summary_vs.pkl"):
            with open("cache/msd_disease_info_summary_vs.pkl","rb") as f:
                self.handbook_vs = pickle.load(f)
        else:
            self.handbook_vs = FAISS.from_texts(texts=[""],embedding=self.embedding_model)

            with open('datasets/medical_knowledgebase_content/handbooks_guideline/msd_disease_info_summary.json') as f:
                tmp_handbook = json.load(f)
                for key in tqdm(tmp_handbook):
                    self.handbook_vs.add_texts([key+":"+str(tmp_handbook[key])],metadatas=[{"key":key}])
            with open("cache/msd_disease_info_summary_vs.pkl","wb") as f:
                pickle.dump(self.handbook_vs,f)
        
        with open('datasets/medical_knowledgebase_content/handbooks_guideline/msd_disease_info.json','r') as f:
            self.disease_info = json.load(f)
        self.retriever = self.handbook_vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})


        self.output_parser = StrOutputParser()


        self.summarize_chain = self.summarize_component |self.chatmodel|self.output_parser
        self.first_round_chain = RunnableParallel({
            "input":RunnablePassthrough()
            })|self.first_round_diagnosis_component|self.chatmodel|self.output_parser

        self.retrieval_chain = self.retriever
        self.second_round_diagnosis_component = ChatPromptTemplate.from_messages([
            ("system", "你是一个医生，根据以下资料和病情或者问题，给出一个诊断或者回答。"),
            ("user", "资料：{input_1},病历描述：{input_2},问题：{input_3}")])

        self.second_round_diagnosis_chain = RunnableParallel({
            "input_1":itemgetter("relevant_docs"),
            "input_2":itemgetter("description"),
            "input_3":itemgetter("input")
            })|self.second_round_diagnosis_component|self.chatmodel
    def invoke(self,input_data,textonly = False):
        if self.verbose:
            print("===输入===")
            print(input_data)
            print("\n")
        x = self.summarize_chain.invoke({"input": input_data})
        if self.verbose:
            print("===病历描述===")
            print(x)
            print("\n")
        x_1:str= self.first_round_chain.invoke(x)
        x_1 = x_1.split(",")
        docs = []
        if self.verbose:
            print("===初步诊断===")
            print(x_1)
            print("\n")
        for item in x_1:
            tmp = self.retrieval_chain.invoke(item)
            docs.append(tmp)
        relevant_docs = ""
        if self.verbose:
            print("===相关资料===")
        docs = docs[:3]
        for doc_list in docs:
            for doc in doc_list:
                try:
                    if self.verbose:
                        print(f"{doc.page_content[:20]}...")
                    name = str(doc.page_content).split(":")[0]
                    relevant_docs+= str(self.disease_info[name])
                except:
                    print(f'Failed to print doc content,original doc content is:{doc.page_content[:20]}...Skip it.')
                    relevant_docs+= " "
        relevant_docs = relevant_docs[:10000]
        if self.verbose:
            print(f"相关资料长度：{len(relevant_docs)}")
            print("\n")
        x_2 = {
            "relevant_docs":relevant_docs,
            "description":x,
            "input":input_data
        }
        y = self.second_round_diagnosis_chain.invoke(x_2)
        y_textonly = self.output_parser.invoke(y)
        if self.verbose:
            print("===Bot诊断===")
            print(y_textonly)
        if textonly:
            return y_textonly
        else:
            return y