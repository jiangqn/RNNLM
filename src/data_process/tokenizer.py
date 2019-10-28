import spacy
import re

url = re.compile('(<url>.*</url>)')
spacy_en = spacy.load('en')


def check(x):
    return len(x) >= 1 and not x.isspace()

def tokenize(text):
    tokens = [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]
    return list(filter(check, tokens))