
from legal_kg import *







if __name__ == '__main__':
    image_name = 'tester'
    text_path = '../input/articles/article4.txt'
    with open(text_path) as f:
        text = f.read()
        tester = LegalKG()
        tester.extract_triples_openie(text, image_name)