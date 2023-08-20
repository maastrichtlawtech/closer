import ast
import os

import openai
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from llama_index import (GPTVectorStoreIndex, LLMPredictor, PromptHelper,
                         ServiceContext, SimpleDirectoryReader, StorageContext,
                         load_index_from_storage)


class OpenaiTask:
    def __init__(self, path, model_name='text-davinci-003',
                 api_key=None, top_p=1,
                 temperature=0, index_type='GPTVectorStoreIndex'):
        self.path = path
        self.model_name = model_name
        self.api_key = api_key
        self.top_p = top_p
        self.temperature = temperature
        self.index_type = index_type
        self.max_input_size = 3000 
        self.num_output = 1000
        self.max_chunk_overlap = 0.1
        self.chunk_size_limit = 512 
        if self.model_name == 'text-davinci-003':
            self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=self.temperature,
                                                         model_name=self.model_name,
                                                         top_p=self.top_p))
        else:
            self.llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=self.temperature,
                                                        model_name=self.model_name,
                                                        top_p=self.top_p
                                                        ))
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
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor,
                                                            prompt_helper=self.prompt_helper)
        self.base_storage_path = os.path.join(self.path, 'storage')
        if api_key is None:
            openai.api_key = os.getenv('OPENAI_API_KEY')
        else:
            openai.api_key = api_key
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
            file_storage_path = self.base_storage_path + '/' + filename_split +'/'
            if filename.endswith(".txt") and not os.path.isdir(file_storage_path):
                if self.index_type == 'GPTVectorStoreIndex':
                    document = SimpleDirectoryReader(
                        input_files=[f"{self.path}/{filename}"]).load_data()
                    index = GPTVectorStoreIndex.from_documents(document)
                    index.index_struct.index_id = filename_split
                    index.storage_context.persist(persist_dir=f"{file_storage_path}") 

    def execute_task(self, article=None, prompt=None):
        if article is None:
            raise ValueError("Please provide an article number to extract from.")
        index = load_index_from_storage(storage_context=StorageContext.from_defaults(
            persist_dir=self.base_storage_path + '/' + article + '/'),
            service_context=self.service_context)
        query_engine = index.as_query_engine()
        full_response = ''
        while True:
            resp = query_engine.query(prompt + '\n\n' + full_response)
            if resp.response != "Empty Response":
                full_response += (" "+ resp.response)
            else:
                break
        print(full_response)
        return self.eval_brokenliteral(full_response)
