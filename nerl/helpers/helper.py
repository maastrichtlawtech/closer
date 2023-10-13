import nltk
import tiktoken
from nltk.tokenize import word_tokenize


class Helper:
    def __init__(self, text, prompt):
        nltk.download('punkt')
        self.text = text
        self.prompts = []
        self.prompt = prompt

    def count_tokens(self):
        tokens = word_tokenize(self.text)
        return len(tokens)

    def break_up_file(self, tokens, chunk_size, overlap_size):
        if len(tokens) <= chunk_size:
            yield tokens
        else:
            chunk = tokens[:chunk_size]
            yield chunk
            yield from self.break_up_file(tokens[chunk_size - overlap_size:],
                                          chunk_size, overlap_size)

    def break_up_file_chunks(self, filename, chunk_size=500, overlap_size=100):
        tokens = word_tokenize(self.text)
        return self.break_up_file(tokens, chunk_size, overlap_size)

    def chunk_text(self):
        token_count = self.count_tokens()
        print("The text {} has {} tokens".format(self.text, token_count))
        chunks = self.break_up_file_chunks(self.text, chunk_size=2000, overlap_size=100)
        for i, chunk in enumerate(chunks):
            print("Chunk {}: {}".format(i, len(chunk)))
            new_prompt = self.prompt + "\n'''" + \
                         self.convert_to_detokenized_text(chunk) + \
                         "\n''' Result:"
            self.prompts.append(new_prompt)
            return self.prompts

    def convert_to_detokenized_text(self, tokenized_text):
        detokenized_text = " ".join(tokenized_text)
        detokenized_text = detokenized_text.replace(" 's", "'s")
        return detokenized_text

    def tiktoken_token(self, model_name):
        if model_name == 'text-davinci-003':
            enc = tiktoken.get_encoding("p50k_base")
        elif model_name == 'gpt-4' or \
                model_name == 'got-3.5-turbo' or \
                model_name == 'text-embedding-ada-002':
            enc = tiktoken.get_encoding("cl100k_base")
        else:
            enc = tiktoken.get_encoding("r50k_base")
        input_text = self.prompt + "\n'''" + self.text + "\n''' Result:"
        enc.encode(input_text)
        num_tokens = len(enc.encode(input_text))
        return num_tokens
