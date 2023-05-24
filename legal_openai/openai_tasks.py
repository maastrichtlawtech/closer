import os

from langchain import OpenAI
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
        self.max_input_size = 4096
        self.num_output = 100
        self.max_chunk_overlap = 20
        self.chunk_size_limit = 600
        self.llm_predictor = LLMPredictor(llm=OpenAI(temperature=self.temperature,
                                                    model_name=self.model_name,
                                                    top_p=self.top_p,
                                                    ))
        self.prompt_helper = PromptHelper(self.max_input_size, self.num_output,
                                          self.max_chunk_overlap,
                                          chunk_size_limit=self.chunk_size_limit)
        self.service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor,
                                                            prompt_helper=self.prompt_helper)
        self.base_storage_path = os.path.join(self.path, 'storage')
        self.load_indices()
        print("Loaded indices")

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
            llm_predictor=self.llm_predictor)
        query_engine = index.as_query_engine()
        full_response = ''
        while True:
            resp = query_engine.query(prompt + '\n\n' + full_response)
            if resp.response != "Empty Response":
                full_response += (" "+ resp.response)
            else:
                break
        return full_response
