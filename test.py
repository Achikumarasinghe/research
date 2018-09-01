import sys
import os
from collections import Counter
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy as np 
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pandas as pd
import re
from string import *
from sklearn.preprocessing import Normalizer
from nltk.tokenize import word_tokenize

corpus = []
stemmer = PorterStemmer()

for file in glob.glob("data/*.txt"):
    with open(file, "r") as doc:
        corpus.append((file, doc.read()))

print(corpus)
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

print(tfidf_matrix)

lsa = TruncatedSVD(n_components=2,n_iter=5)
lsa.fit(tfidf_matrix)
print(lsa)

terms = tf.get_feature_names()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")


query = 'MONEY MARKET TRADING STOCKS INVESTMENTS BULL COMPANY INDUSTRY VALUE PRICES'
tokens = word_tokenize(query)
print("tokens :")
print(tokens)


qtf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')

q_tfidf_matrix =  qtf.fit_transform(tokens)

print("qtfidf:")
print(q_tfidf_matrix)

#lsa = TruncatedSVD(n_components=20,n_iter=5)
#lsa.fit(q_tfidf_matrix)

# Euclidean distance between two vectors
def distance(v1, v2):
	return np.linalg.norm(v1-v2)

# representation of the query and the representations of the documents
#dist = [distance(q_tfidf_matrix,d) for d in tfidf_matrix.T]

#print("disntance :")
#print(dist)