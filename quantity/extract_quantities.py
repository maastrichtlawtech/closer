import os

from dotenv import load_dotenv
from legal_openai.openai_tasks import OpenaiTask
from quantulum3 import parser

load_dotenv()
prompt_path = os.getenv('PROMPT_PATH')

class QuantitiesExtractor:
    def __init__(self):
        pass

    def quantulum_extract(self, text):
        quants = parser.parse(text)
        temp_list = []
        for quantity in quants:
            if quantity.unit.name == 'dimensionless':
                continue
            if quantity.surface is not None:
                temp_list.append(quantity.surface)
        return temp_list

    def openai_extract(self, prompt=None, article=None, path=None, api_key=None):
        if prompt is None:
            with open(prompt_path + 'quantity.txt', 'r') as f:
                prompt = f.read()
        return OpenaiTask(path=path, api_key=api_key).execute_task(
            prompt=prompt, article=article)

