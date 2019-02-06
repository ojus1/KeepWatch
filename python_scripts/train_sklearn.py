from sklearn.tree import DecisionTreeClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn_porter import Porter
import pandas as pd
import numpy as np
import json
import nltk
nltk.download('punkt')

charLenLim = 9

dftrain = pd.read_csv("../data/sample_data.csv")
#dftrain = dftrain.sample(n=14900//3, axis=0)
urls = dftrain['url']

with open("../data/corpus_set.txt", 'r') as f:
    corpus_set = f.read().split(" ")


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
    return np.array(list(global_bow.values()))

labels = dftrain['label']
del dftrain
X = np.array([extract_features(url) for url in urls])
print("Prepared X, shape: ", X.shape)
#print(X)
del urls
labels = pd.get_dummies(labels, columns=['label'], drop_first=True)
y = labels.values
print("Prepared y, shape: ", y.shape)
#print(y)
del labels

#xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.3)

model = DecisionTreeClassifier()
model.fit(X, y)

#scores = cross_val_score(model, X, y, cv=3)
scores = model.score(X, y)
print("Train accuracy:", scores)

porter = Porter(model, language='js')
output = porter.export(embed_data=True)

#print(output)
with open("../transpiled_predict_js.js", "w") as f:
    f.write(output)
