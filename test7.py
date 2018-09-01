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


corpus = []
stemmer = PorterStemmer()

for file in glob.glob("testData/*.txt"):
    with open(file, "r") as doc:
        corpus.append((file, doc.read()))

print(corpus)
print("= ="*20)

tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')
tfidf_matrix =  tf.fit_transform([content for file, content in corpus])
print("TFIDF\n",pd.DataFrame(tfidf_matrix.toarray()))
print("= ="*20)

tfidf_matrix_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tf.get_feature_names())
print("TFIDF WITH FEATURE NAMES\n",tfidf_matrix_df)
print("= ="*20)

cos_sim_matix_tfidf =(tfidf_matrix * tfidf_matrix.T).toarray()
cos_sim_matix_tfidf_df = pd.DataFrame(cos_sim_matix_tfidf)
print("COS SIM MATRIX TFIDF\n",cos_sim_matix_tfidf_df)
print("= ="*20)

svd =  TruncatedSVD(n_components=2,n_iter=5)
lsa = svd.fit_transform(tfidf_matrix.T)
lsa_df = pd.DataFrame(lsa)
print("LSA\n",lsa_df)
print("= ="*20)

cos_sim_matix = np.dot(lsa,lsa.T)
cos_sim_matix_df = pd.DataFrame(cos_sim_matix)
print("COS SIM MATRIX LSA\n",cos_sim_matix_df)
print("= ="*20)


index = tf.vocabulary_["depend"]
print("index of depend is : ",index)

print("= ="*20)
print("WORD TO WORD SIMILARITY")
print(np.concatenate((cos_sim_matix[index][:index] , cos_sim_matix[index][(index+1):])))

print("= ="*20)
print("MAX SIMILARITY")
print(np.argmax(np.concatenate((cos_sim_matix[index][:index] , cos_sim_matix[index][(index+1):])))) 




