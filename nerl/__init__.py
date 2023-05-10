'''
import os

from dotenv import load_dotenv

from nerl import EntityRecognizer

load_dotenv()
tagme_api_key = os.getenv("GCUBE_TOKEN")
openai_api_key = os.getenv("OPENAI_API_KEY")
entity = EntityRecognizer()
text = "I am going to London next week. After that, I am going to New York"
# Uses refined = 
# print(entity.spacy_recognize(text))
# print(entity.eurovoc_recognize(text))
# print(entity.wikipedia_tagme(text, tagme_api_key, threshold=0.1))
# print(entity.refined_recognise(text))
print(openai_api_key)
entity.openai_recognise(text=text, prompt="You are an entity recogniser. For the given text, you need to find the entities. This is the text and you will return the results in the format {'entities': [{'text': 'London', 'uri': 'http://www.wikidata.org/wiki/Q84'}, {'text': 'New York', 'uri': 'http://www.wikidata.org/wiki/Q60'}]]}",
                        api_key=openai_api_key)
'''
