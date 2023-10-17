from extract_quantities import *

if __name__ == '__main__':
    image_name = 'tester'
    text_path = '../input/articles/article4.txt'
    with open(text_path) as f:
        text = f.read()
        extractor = QuantitiesExtractor()
        print(extractor.quantulum_extract(text))