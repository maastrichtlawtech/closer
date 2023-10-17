import json
import os

import nltk
from dotenv import load_dotenv

from nerl.nerl import EntityRecognizer
from parsers.eurlex import get_text_from, process_text
from quantity.extract_quantities import QuantitiesExtractor
from references.extract_references import ExtractReferences
from rule_classification.deontic_logic import DeonticLogic

load_dotenv()
tagme_api_key = os.getenv("GCUBE_TOKEN")
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialise dependencies
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()


if __name__ == '__main__':
    # Recognise entities in the text
    # By iterating through all articles in the articles folder
    # Or parse it from the html link
    html_link = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0947&from=EN"
    legal_text = get_text_from(html_link)

    # Read schema for data storage
    with open('./input/schema.json', 'r') as f:
        json_schema = json.load(f)

    # Add more structure to the schema
    json_schema["document_data"] = {}
    json_schema["document_data"]["uri"] = html_link
    json_schema["document_data"]["sections"] = {}
    json_schema["document_data"]["sections"]["chapters"] = []

    # Process the text
    processed_legal_text = process_text(legal_text)

    # Keep track of articles. Temp setup until article number, name are extracted
    # in helper functions.
    article_count = 0
    articles = []

    # Initialise entity recogniser object
    entity = EntityRecognizer()
    for article in os.listdir('./input/articles/'):
        article_split = article.split('.txt')[0]
        print(f"Processing article {article_split}")
        if article.endswith('.txt'):
            with open('./input/articles/' + article, 'r') as f:
                text = f.read()
            article_collection = {}
            article_count += 1
            article_collection["id"] = article_count
            article_collection["text"] = text
            article_collection["uri"] = ""
            sent_collection = []
            try:
                for sent in tokenizer.tokenize(text):
                    temp_sent_data = {}
                    temp_sent_data["text"] = sent
                    classification_data = DeonticLogic(path='./input/articles/'). \
                        openai_classifier_logic_only(api_key=openai_api_key,
                                                     article=article_split, text=sent)
                    temp_sent_data["sentence_rule"] = classification_data['class']
                    temp_sent_data["obligation"] = classification_data['obligation']
                    temp_sent_data["obligation_holder"] = classification_data['obligation_holder']
                    temp_sent_data["right_holder"] = classification_data['right_holder']
                    temp_sent_data["permission"] = classification_data['permission']
                    temp_sent_data["prohibition"] = classification_data['prohibition']
                    temp_sent_data["bearer"] = classification_data['bearer']
                    temp_sent_data["omission"] = classification_data['omission']
                    temp_sent_data["concepts"] = entity.spacy_recognize(sent)
                    temp_sent_data["references"] = ExtractReferences().openai_references(
                        path='./input/articles/', api_key=openai_api_key, article=article_split)
                    temp_sent_data["quantity"] = QuantitiesExtractor().quantulum_extract(text)
                    temp_sent_data["if_then"] = DeonticLogic(path='./input/articles/'). \
                        openai_if_then_fetch(api_key=openai_api_key,
                                             article=article_split, text=sent)
                    temp_sent_data["describe_if_then"] = DeonticLogic(path='./input/articles/'). \
                        openai_if_then_describe(api_key=openai_api_key,
                                                article=article_split, text=sent)
                    sent_collection.append(temp_sent_data)
                    article_collection["sentences"] = sent_collection
                articles.append(article_collection)
            except Exception as e:
                print(f"error processing, moving on to the next one {e}")
                pass
        json_schema["document_data"]["sections"]["articles"] = articles
        print(json_schema)

    with open("output/processed_data.json", "w", encoding="utf-8") as jsonFile:
        json.dump(json_schema, jsonFile, ensure_ascii=False, indent=4)
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
    print(ExtractReferences().openai_references(path='./input/articles/',
                                                api_key=openai_api_key, article='article4'))
    print(DeonticLogic(path='./input/articles/').openai_classifier(api_key=openai_api_key,
                                                                   article='article5'))
    '''
