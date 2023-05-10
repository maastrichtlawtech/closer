import os

from dotenv import load_dotenv

from nerl.nerl import EntityRecognizer
from references.extract_references import ExtractReferences

load_dotenv()
tagme_api_key = os.getenv("GCUBE_TOKEN")
openai_api_key = os.getenv('OPENAI_API_KEY')
# Recognise entities in the text
# By iterating through all articles in the articles folder
with open('./prompts/rule_classification_prompt.txt', 'r') as f:
    classification_prompt = f.read()
entity = EntityRecognizer()
'''
for article in os.listdir('./input/articles/'):
    article_split = article.split('.txt')[0]
    if article.endswith('.txt'):
        with open('./input/articles/' + article, 'r') as f:
            text = f.read()
        #print(entity.spacy_recognize(text))
        #print(entity.eurovoc_recognize(text, tsv_file='./nerl/input/eurovoc.tsv'))
        #print(entity.wikipedia_tagme(text, tagme_api_key, threshold=0.2))
        #print(entity.refined_recognise(text))
        print(entity.openai_recognise(openai_api_key, article=article_split))
'''
print(ExtractReferences().openai_references(path='./input/articles/',
                                            api_key=openai_api_key, article='article4'))
'''
print(DeonticLogic(path='./input/articles/').openai_classifier(api_key=openai_api_key,
                                                               article='article5'))
'''
