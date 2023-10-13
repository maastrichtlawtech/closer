import logging
import sys

import pandas as pd
from openie import StanfordOpenIE

logger = logging.getLogger(__name__)


class LegalKG:
    def __init__(self):
        self.properties = {
            'openie.affinity_probability_cap': 2 / 3,
        }

    def extract_triples_openie(self, text=None,
                               image_name=None,
                               image_path='./output/'):
        if text is None:
            logging.exception("No text to extract triples from")
            sys.exit(1)
        triple_list = []
        with StanfordOpenIE(properties=self.properties) as client:
            for triple in client.annotate(text):
                logger.info('|-' + str(triple))
                triple_list.append(triple)
            if image_name is None:
                logger.info("Image name not provided")
            else:
                client.generate_graphviz_graph(text, image_path + image_name + '.png')
                logger.info('Graph generated:' + image_path + image_name + '.png')
        return triple_list

    # Convert triples to RDF
    def convert_triples_to_rdf(self, triple_list=None, entities=None):
        # Map entities and triples_list and combine them with subject(s) and object(s) having URIs
        df = pd.DataFrame(columns=['subject', 'subject_uri', 'predicate', 'object', 'object_uri'])
        for triple in triple_list:
            logger.info(triple)
            for key, value in entities.items():
                logger.info('key:' + key + ' value:' + str(value))
                if triple['subject'].lower().find(key.lower()):
                    df['subject_uri'] = '<' + str(value) + '>'
                    df['subject'] = triple['subject']
                    df['predicate'] = triple['relation']
                if key.lower() in triple['object'].lower():
                    df['object_uri'] = '<' + str(value) + '>'
                    df['object'] = triple['object']
        logger.info(df)
        return df
