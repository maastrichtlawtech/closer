import json
import logging
import os

import rdfpandas
from dotenv import load_dotenv

from kg_creation.legal_kg import LegalKG
from nerl.nerl import EntityRecognizer

logger = logging.getLogger(__name__)
load_dotenv()
tagme_api_key = os.getenv("GCUBE_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ['TOKENIZERS_PARALLELISM'] = "false"
# Set EU legislation document link 
eu_legislation_2019_947_link = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=CELEX:32019R0947&from=EN"

# Import schema for data storage
with open('./input/schema.json', 'r') as f:
    json_schema = json.load(f)

# Add more structure to the schema
json_schema["document_data"] = {}
json_schema["document_data"]["uri"] = eu_legislation_2019_947_link
json_schema["document_data"]["sections"] = {}
json_schema["document_data"]["sections"]["articles"] = []

'''
# Document parsing
eu_legislation_2019_947_text = get_text_from(eu_legislation_2019_947_link)
eu_legislation_2019_947_processed_text = process_text(eu_legislation_2019_947_text)

# If parsed document needs to be saved in txt file
for key in eu_legislation_2019_947_processed_text["order"]:
    if not os.path.isdir('./input/eu_legislation_2019_947'):
        os.mkdir('./input/eu_legislation_2019_947')
    else:
        with open('./input/eu_legislation_2019_947/' + 'eu_2019_947.txt', 'a') as f:
            if key != 'articles':
                f.write(str(eu_legislation_2019_947_processed_text[key][0]))
            else:
                for art in eu_legislation_2019_947_processed_text[key]:
                    f.write(str(art))
        # Store sections of the document individually in txt files for further processing
        if not os.path.isdir('./input/eu_legislation_2019_947/document_split'):
            os.mkdir('./input/eu_legislation_2019_947/document_split')
        else:
            if key == 'articles':
                count = 1
                for i in eu_legislation_2019_947_processed_text[key]:
                    with open('./input/eu_legislation_2019_947/document_split/' + 'article_'+ str(count) + '.txt', 'w') as f:
                        f.write(str(i))
                    count += 1
            else:
                with open('./input/eu_legislation_2019_947/document_split/' + key + '.txt', 'w') as f:
                    f.write(str(eu_legislation_2019_947_processed_text[key][0]))
'''
# Perform necessary tasks; we do so only for Article 7 and 9 for illustration purposes
entity = EntityRecognizer()
for article in os.listdir('./input/eu_legislation_2019_947/document_split/'):
    article_split = article.split('.txt')[0]
    if article_split == 'article_7' or article_split == 'article_9':
        with open('./input/eu_legislation_2019_947/document_split/' + article, 'r') as f:
            article_text = f.read()
        total_count = {}
        # Perform entity recognition with various methods
        '''
        temp_entity_data_spacy = entity.spacy_recognize(article_text)
        logger.info("Output for spacy:\n" + str(temp_entity_data_spacy))
        '''
        temp_entity_data_tagme = entity.wikipedia_tagme(article_text, tagme_api_key,
                                                        threshold=0.2)
        logger.info("Output for tagme:\n" + str(temp_entity_data_tagme))
        '''
        temp_entity_data_eurovoc = entity.eurovoc_recognize(article_text,
                                                            tsv_file='./nerl/input/eurovoc.tsv')
        logger.info("Output for eurovoc:\n" + str(temp_entity_data_eurovoc))
    
        temp_entity_data_refined = entity.refined_recognize(article_text)
        logger.info("Output for refined:\n" + str(temp_entity_data_refined))
        temp_entity_data_openai_recognise = OpenAiNew().get_response(filename=article_split,
                                                                    path='./input/eu_legislation_2019_947/document_split',
                                                                    task_type="entity_recognition") 
        logger.info("Output for openai:\n" + str(temp_entity_data_openai_recognise))
        # Calculate the number of entities recognized by each method
        total_count['spacy_entity'] = len(temp_entity_data_spacy)
        total_count['tagme_entity'] = len(temp_entity_data_tagme)
        total_count['eurovoc_entity'] = len(temp_entity_data_eurovoc)
        total_count['refined_entity'] = len(temp_entity_data_refined)
        total_count['openai_entity_normal_prompt'] = len(temp_entity_data_openai_recognise)
        logger.info(total_count)
        df = pd.DataFrame(total_count.items())
        df.to_csv(f'./output/task_details_for_{article_split}.csv', index=False)
        # Peform condition identification
        temp_condition_data_openai_identify = OpenAiNew().get_response(filename=article_split,
                                                                       path='./input/eu_legislation_2019_947/document_split',
                                                                       task_type="condition_identification")
        logger.info("Output for openai:\n" + str(temp_condition_data_openai_identify))
        # Calculate the number of conditions identified
        total_count['openai_condition_normal_prompt'] = len(temp_condition_data_openai_identify)
        logger.info(total_count)
        df = pd.DataFrame(total_count.items())
        df.to_csv(f'./output/task_details_for_{article_split}.csv',
                  index=False)
        
        # Perform deontic modality extraction
        temp_deontic_data_openai_identify = OpenAiNew().get_response(filename=article_split,
                                                                     path='./input/eu_legislation_2019_947/document_split',
                                                                     task_type="deontic_modality")
        # Calculate the number of modalities identified
        total_count['openai_deontic_normal_prompt'] = len(temp_deontic_data_openai_identify)
        logger.info(total_count)
        df = pd.DataFrame(total_count.items())
        df.to_csv(f'./output/task_details_for_{article_split}.csv',
                  index=False)
        '''
        # Perform triple extraction
        temp_triple_data_openie = LegalKG().extract_triples_openie(text=article_text,
                                                                   image_name=article_split)
        keys = temp_triple_data_openie[0].keys()
        '''
        with open('./output/triples_of_' + article_split + '.csv', 'w') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(temp_triple_data_openie)
        '''
        # Convert triples to knowledge graph
        temp_kg_data = LegalKG().convert_triples_to_rdf(triple_list=temp_triple_data_openie,
                                                        entities=temp_entity_data_tagme)
        if not temp_kg_data.empty:
            # Store the knowledge graph in rdf file
            g = rdfpandas.to_graph(temp_kg_data)
            ttl = g.serialize(format='turtle')
            with open('./output/article_' + article_split + '_kg.ttl', 'wb') as f:
                f.write(ttl)
        else:
            logger.info("No knowledge graph can be created for " + article_split)
        # Store all details in json file for downstream application
