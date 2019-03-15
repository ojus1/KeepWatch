from sklearn.metrics import confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')

dftrain = pd.read_csv("../data/sample_data.csv")
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



X = np.array([extract_features(url) for url in urls])
print("Prepared X, shape: ", X.shape)
#print(X)
del urls
labels = dftrain.replace({"bad": 1, "good": 0})['label']
del dftrain
y = labels.values
print("Prepared y, shape: ", y.shape)
#print(y)
del labels

xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.3)

model = ExtraTreesClassifier()
model.fit(xtr, ytr)

#scores = cross_val_score(model, X, y, cv=3)
scores = model.score(xts, yts)
print("Validation score:", scores)

y_pred = model.predict(xts)
# Compute confusion matrix
cnf_matrix = confusion_matrix(yts, y_pred)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign', 'Malicious'], title='Confusion matrix')

plt.show()