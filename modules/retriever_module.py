import os
import json
import hashlib
from tqdm import tqdm
import logging
import shutil

from logging import Logger
from langchain_community.document_loaders import UnstructuredPDFLoader,UnstructuredFileLoader#,JSONLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from rapidocr_onnxruntime import RapidOCR
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pdf_loader import RapidOCRPDFLoader


logger = logging.getLogger(__name__)


class RetrieverModule():
    def __init__(self,embedding_model:str='BAAI--bge-large-zh-v1.5',db_root:str='knowledge_base',db_name:str='Main',data_dir:str='MedData'):
        
        self.db_root = db_root
        self.db_name = db_name
        self.embedding_model_name = embedding_model
        self.embedding_model = HuggingFaceBgeEmbeddings(model_name=os.path.join('models','embedding_models',embedding_model))
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128,separators=["\n\n", "\n", " ", ""])
        if not os.path.exists(os.path.join(db_root,db_name)):
            self._create_db()
            self.vectorstore = FAISS.from_texts(texts=[""],embedding=self.embedding_model)
            self.vectorstore.save_local(os.path.join(db_root,db_name,'vector_store',self.embedding_model_name))
        else:
            self.vectorstore = FAISS.load_local(os.path.join(db_root,db_name,'vector_store',self.embedding_model_name),embeddings=self.embedding_model)
        self.db_json = self._reload_db_json()
        
    def _create_db(self):
        os.makedirs(os.path.join(self.db_root,self.db_name))
        os.makedirs(os.path.join(self.db_root,self.db_name,'content'))
        os.makedirs(os.path.join(self.db_root,self.db_name,'vector_store'))
        os.makedirs(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name))
        new_flag = True
        with open (os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'w') as f:
            f.write('{}')
    def _get_loader(self,file_path:str):
        class TXTloader():
            def __init__(self,file_path):
                self.file_path = file_path
            def load(self):
                with open(self.file_path,'r') as f:
                    return f.read()
        class JSONLoader():
            def __init__(self,file_path):
                self.file_path = file_path
            def load(self):
                with open(self.file_path,'r') as f:
                    data = json.load(f)
                result = []
                for key in data:
                    result.append(f"{key}:{data[key]}")
                result = "\n\n".join(result)
                return result
        if file_path.endswith('.pdf'):
            return RapidOCRPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            return UnstructuredFileLoader(file_path)
        elif file_path.endswith('.json'):
            return JSONLoader(file_path)
        else:
            raise ValueError(f'Unknown file type: {file_path}')
    def _reload_db_json(self):
        with open(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'r') as f:
            self.db_json:dict = json.load(f)        
    def add_files(self,doc_path_list:list):
        self._reload_db_json()
        
        for file_path in tqdm(doc_path_list):
            filename = os.path.basename(file_path)
            if filename in self.db_json.keys():
                print(f'File: {file_path} already exists in db')
            else:
                print("copying...")
                shutil.copy(file_path,os.path.join(self.db_root,self.db_name,'content',os.path.basename(file_path)))
                print("loading...")
                loader = self._get_loader(file_path)
                doc = loader.load()
                for idx,page in enumerate(doc):
                    print("splitting...")
                    page_piece_content_list = self.text_splitter.split_text(page.page_content)
                    page_piece_metadata_list = []
                    for page_piece in page_piece_content_list:
                        page_piece_metadata_list.append(
                            {
                                'page':idx+1,
                                'source':filename,
                            }
                        )
                    print("adding...")
                    id = self.vectorstore.add_texts(texts=page_piece_content_list,metadatas=page_piece_metadata_list)
                print(f'Add file: {file_path} with id: {id}')
                self.db_json[filename]=1
        self.vectorstore.save_local(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name))
        with open(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'w') as f:
            json.dump(self.db_json,f,ensure_ascii=False,indent=4)
    def get_files(self):
        self._reload_db_json()
        return self.db_json.keys()
    def delete_files(self,filenames:list):
        self._reload_db_json()
        for filename in filenames:
            if filename not in  self.db_json.keys():
                print(f'File: {filename} not exists in db')
                continue
            ids = []
            # for k, v in self.vectorstore.docstore._dict.items():
            #     if v.metadata.get("source") == filename:
            #         ids.append(k)
            ids = [k for k, v in self.vectorstore.docstore._dict.items() if v.metadata.get("source") == filename]

            self.vectorstore.delete(ids)
            os.remove(os.path.join(self.db_root,self.db_name,'content',filename))
            self.db_json.pop(filename)
        self.vectorstore.save_local(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name))
        with open(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'w') as f:
            json.dump(self.db_json,f,ensure_ascii=False,indent=4)
            
            
    # def add_and_update_db(self,doc_path_list:list):
    #     with open(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'r') as f:
    #         last_list = json.load(f)
    #     add_list = []
    #     delete_list = []
    #     for root, dirs, files in os.walk(doc_path_list):
    #         # calc hash
    #         for file in files:
    #             if file.endswith('.pdf'):
    #                 filename_list = [item['path'] for item in last_list]
    #                 src_file = os.path.join(root, file)
    #                 if src_file not in filename_list:
    #                     add_list.append({
    #                         'path':src_file,
    #                         'hash':hashlib.md5(open(src_file, 'rb').read()).hexdigest()
    #                     })
    #                 else:
    #                     for item in last_list:
    #                         if item['path'] == src_file:
    #                             with open(src_file, 'rb') as f:
    #                                 new_hash = hashlib.md5(f.read()).hexdigest()
    #                             if new_hash != item['hash']:
    #                                 add_list.append({
    #                                     'path':src_file,
    #                                     'hash':new_hash
    #                                 })
    #                                 delete_list.append(item)
        
    #     print('add_list:',add_list)
    #     print('delete_list:',delete_list)
    #     for item in delete_list:
    #         self.vectorstore.delete(item['id'])#TODO: It's not working currently
    #         os.remove(item['path'])
    #         last_list.remove(item)
    #     for item in add_list:
    #         docs = self.load_pdf_file(pdf_file = item['path'])
    #         docs = self.text_splitter.split_documents(docs)
    #         id = self.vectorstore.add_documents(docs)
    #         last_list.append({
    #             'path':item['path'],
    #             'hash':item['hash'],
    #             'id':id,
    #         })
    #     with open(os.path.join(self.db_root,self.db_name,'vector_store',self.embedding_model_name,'db_list.json'),'w') as f:
    #         json.dump(last_list,f)
    #     return add_list,delete_list


    # def load_pdf_file(self,pdf_file):    
    #     loader = RapidOCRPDFLoader(pdf_file)
    #     docs = loader.load()
    #     name = os.path.basename(pdf_file)
    #     with open(os.path.join('tmp',name+'.txt'),'w') as f:
    #         f.write(docs[0].page_content)
    #     print('pdf:\n',docs[0].page_content[:100])
    #     return docs
