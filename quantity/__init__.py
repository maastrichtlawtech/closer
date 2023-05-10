from extract_quantities import QuantitiesExtractor

with open('../rule_classification/articles/article4.txt', 'r') as f:
    text = f.read()

quantities = QuantitiesExtractor(text).quantulum_extract()
print(quantities)
