from sklearn_porter import Porter
import pandas as pd
import numpy as np
import json
import nltk
nltk.download('punkt')

charLenLim = 12

dftrain = pd.read_csv("../data/sample_data.csv")
#dftrain = dftrain.sample(n=14900//10, axis=0)
urls = dftrain['url']
del dftrain

corpus = " ".join(urls).lower()

dftrain = pd.read_csv("../data/data2.csv")
#dftrain = dftrain.sample(n=14900//10, axis=0)
urls = dftrain['url']
del dftrain

corpus = corpus + " ".join(urls).lower()
pattern = r'[a-z]+|[A-Z]+|\W+'

corpus2 = nltk.regexp_tokenize(corpus, pattern=pattern)
corpus_set = sorted(list(set(corpus2)))
corpus_set = sorted([item for item in corpus_set if len(item) < charLenLim])

#with open("../data/corpus_set.txt", 'w') as f:
#    f.write(" ".join(corpus_set))

def count(tkn, tokens):
    c = 0
    for token in tokens:
        if tkn == token:
            c += 1
    return c

def extract_features(text):
    pattern = r'[a-z]+|[A-Z]+|\W+'
    tokens = nltk.regexp_tokenize(text, pattern=pattern)

    counts = list()
    for token in list(set(tokens)):
        counts.append(count(token, tokens))
    
    local_bow = {k:v for k, v in zip(list(set(tokens)), counts)}
    global_bow = dict()
    for tk in corpus_set:
        if tk in local_bow.keys():
            global_bow[tk] = local_bow[tk]
        elif tk not in local_bow.keys():
            global_bow[tk] = 0
    return global_bow

frequency_dict = extract_features(corpus)
del corpus
print(len(frequency_dict))

final_corpus_set = list()
for k, v in frequency_dict.items():
    if v > 2:
        final_corpus_set.append(k)
print(len(list(set(final_corpus_set))))

with open("../data/corpus_set.txt", 'w') as f:
    f.write(" ".join(final_corpus_set))
