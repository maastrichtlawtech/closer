import ast
import logging
import os
import sys

import openai
from llama_index import (PromptHelper, ServiceContext, SimpleDirectoryReader,
                         StorageContext, VectorStoreIndex,
                         load_index_from_storage)
from llama_index.llms import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
openai.log = "info"


class OpenaiTask:
    def __init__(self, path, model='text-davinci-003',
                 api_key=None, top_p=1,
                 temperature=0, index_type='VectorStoreIndex',
                 use_index=True):
        self.path = path
        self.model = model
        self.api_key = api_key
        self.top_p = top_p
        self.temperature = temperature
        self.index_type = index_type
        self.max_input_size = 3000
        self.num_output = 1000
        self.max_chunk_overlap = 0.1
        self.chunk_size_limit = 512
        self.use_index = use_index
        self.llm = OpenAI(temperature=self.temperature,
                          model=self.model,
                          top_p=self.top_p)
        '''
        self.prompt_helper = PromptHelper(context_window=self.max_input_size,
                                         chunk_overlap_ratio=self.max_chunk_overlap,
                                         chunk_size_limit=self.chunk_size_limit,
                                          num_output=self.num_output)
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor,
                                                            prompt_helper=self.prompt_helper,
                                                           chunk_size_limit=self.chunk_size_limit)
        '''
        self.prompt_helper = PromptHelper()
        self.service_context = ServiceContext.from_defaults(llm=self.llm,
                                                            prompt_helper=self.prompt_helper)
        self.base_storage_path = os.path.join(self.path, 'storage')
        if api_key is None:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            openai.api_key = api_key
        if self.use_index:
            print(f"{self.path, self.base_storage_path}")
            self.load_indices()
            print("Loaded indices")

    def eval_brokenliteral(self, value_obj, default_value=None, print_error=True):
        # https://stackoverflow.com/questions/75503925/how-to-extract-incomplete-python-objects-from-string
        try:
            return ast.literal_eval(value_obj)
        except SyntaxError as se:
            eval_error = se
            bracket_pairs = {'{': '}', '[': ']'}
            closers, cur_closer = [], ''
            for c in ''.join(value_obj.split()):
                if c not in bracket_pairs:
                    break
                cur_closer = bracket_pairs[c] + cur_closer
                closers.append(cur_closer)
            for closer in closers:
                sub_str = value_obj.strip()
                while sub_str[1:]:
                    try:
                        return ast.literal_eval(sub_str + closer)
                    except SyntaxError:
                        sub_str = sub_str[:-1].strip()
            if print_error:
                print(f"Error: {repr(eval_error)}")
            return default_value

    def load_indices(self):
        for filename in os.listdir(self.path):
            filename_split = filename.split(".txt")[0]
            file_storage_path = self.base_storage_path + '/' + filename_split + '/'
            if filename.endswith(".txt") and not os.path.isdir(file_storage_path):
                if self.index_type == 'VectorStoreIndex':
                    document = SimpleDirectoryReader(
                        input_files=[f"{self.path}/{filename}"]).load_data()
                    index = VectorStoreIndex.from_documents(document)
                    index.index_struct.index_id = filename_split
                    index.storage_context.persist(persist_dir=f"{file_storage_path}")

    def execute_task(self, article=None, prompt=None):
        if article is None:
            raise ValueError("Please provide an article number to extract from.")
        if self.use_index:
            index = load_index_from_storage(storage_context=StorageContext.from_defaults(
                persist_dir=self.base_storage_path + '/' + article + '/'),
                service_context=self.service_context)
            query_engine = index.as_query_engine()
            full_response = ''
            while True:
                resp = query_engine.query(prompt + '\n\n' + full_response)
                if resp.response != "Empty Response":
                    full_response += (" " + resp.response)
                else:
                    break
            print(full_response)
        else:
            documents = SimpleDirectoryReader(input_files=[f"{self.path}/{article}.txt"]).load_data()
            index = VectorStoreIndex.from_documents(documents)
            query_engine = index.as_query_engine()
            full_response = ''
            while True:
                resp = query_engine.query(prompt + '\n\n' + full_response)
                print(resp.response)
                if resp.response != "Empty Response":
                    full_response += (" " + resp.response)
                else:
                    break
            print(full_response)
            return self.eval_brokenliteral(full_response)
