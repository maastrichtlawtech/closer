import ast
import logging
import os
import sys

import openai
from dotenv import load_dotenv
from llama_index import (KnowledgeGraphIndex, PromptHelper, ServiceContext,
                         SimpleDirectoryReader, StorageContext,
                         VectorStoreIndex, load_index_from_storage)
from llama_index.graph_stores import SimpleGraphStore
from llama_index.llms import OpenAI

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logger = logging.getLogger(__name__)

load_dotenv()
openai.log = "info"


class OpenAiNew:
    def __init__(self):
        openai.api_key = os.environ["OPENAI_API_KEY"]
        self.prompt_path = os.environ["PROMPT_PATH"]

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

    def get_response(self, filename=None, path=None, prompt=None, task_type=None, store_index=True):
        if filename is None:
            logger.exception("Filename not provided to process, exiting")
            sys.exit(1)
        if path is None:
            logger.exception("Path not provided to process, exiting")
            sys.exit(1)
        if prompt is None:
            if task_type == 'entity_recognition':
                with open(self.prompt_path + 'normal_prompts/entity_recognition.txt', 'r') as f:
                    prompt = f.read()
            if task_type == 'condition_identification':
                with open(self.prompt_path + 'normal_prompts/if_then.txt', 'r') as f:
                    prompt = f.read()
            if task_type == 'deontic_modality':
                with open(self.prompt_path + 'normal_prompts/deontic_modality.txt', 'r') as f:
                    prompt = f.read()
        llm = OpenAI(temperature=0, model='text-davinci-003', top_p=1.0)
        prompt_helper = PromptHelper()
        service_context = ServiceContext.from_defaults(llm=llm,
                                                       prompt_helper=prompt_helper)
        if not os.path.exists(f"{path}/storage/{filename}"):
            documents = SimpleDirectoryReader(input_files=[f"{path}/{filename}.txt"]).load_data()
            if task_type == 'kg_creation':
                graph_store = SimpleGraphStore()
                storage_context = StorageContext.from_defaults(graph_store=graph_store)
                index = KnowledgeGraphIndex.from_documents(documents,
                                                           max_triplets_per_chunk=3,
                                                           storage_context=storage_context,
                                                           service_context=service_context)
            else:
                index = VectorStoreIndex.from_documents(documents, service_context=service_context)
            index.index_struct.index_id = filename
            index.storage_context.persist(persist_dir=f"{path}/storage/")
        else:
            index = load_index_from_storage(storage_context=StorageContext.from_defaults(
                persist_dir=path + '/storage' + filename + '/'),
                service_context=service_context)
        if task_type == 'kg_creation':
            with open(f"{path}/{filename}.txt") as f:
                text = f.read()
                index = KnowledgeGraphIndex(max_triplets_per_chunk=3)
                triples = index._extract_triplets(text)
                logger.info(triples)
                return triples
        else:
            query_engine = index.as_query_engine()
            response = query_engine.query(prompt)
            logger.info(response.response)
            return self.eval_brokenliteral(response.response)
