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
import subprocess
from graphviz import Source

#file = open("data/text.txt","r") 
#print(file.read())

 #for doc in file.readlines():
 #   tf = Counter()
  #  for word in doc.split():
  #      tf[word] +=1
  #  print(tf.items())

corpus = []
stemmer = PorterStemmer()

for file in glob.glob("data/*.txt"):
    with open(file, "r") as doc:
        corpus.append((file, doc.read()))

print(corpus)
tf = TfidfVectorizer(analyzer='word', min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

print(tfidf_matrix)

#lsa = TruncatedSVD(n_components=20,n_iter=5)
#lsa.fit(tfidf_matrix)
svd =  TruncatedSVD(n_components=2,n_iter=5)
#svd = TruncatedSVD(n_components = 2, algorithm=&amp;quot;arpack&amp;quot;)
lsa = svd.fit_transform(tfidf_matrix.T)
#terms = tf.get_feature_names()

#for i,comp in enumerate(lsa.components_):
#    termsInComp = zip(terms,comp)
#    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
#   print("Concept %d:" % i)
#    for term in sortedterms:
#        print(term[0])
#    print(" ")

#query = 'MONEY MARKET TRADING STOCKS INVESTMENTS BULL COMPANY INDUSTRY VALUE PRICES'

def getClosestTerm(term,transformer,model):
 
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]      
 
    model = np.dot(model,model.T)
    searchSpace =np.concatenate( (model[index][:index] , model[index][(index+1):]) )  
 
    out = np.argmax(searchSpace)
 
    if out<index:
        return transformer.get_feature_names()[out]
    else:
        return transformer.get_feature_names()[(out+1)]
 
def kClosestTerms(k,term,tf,model):
 
    term = stemmer.stem(term)
    index = tf.vocabulary_[term]
 
    model = np.dot(model,model.T)
 
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[tf.get_feature_names()[i]] = model[index][i]
 
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
 
    return sortedList[::-1][0:k]

print("= ="*20)
print("closet term for yield is : ")
print(getClosestTerm("yield",tf,lsa))
print("= ="*20)
print("5 closet terms for yield is : ")
print(kClosestTerms(6,"yield",tf,lsa))
print("= ="*20)
print("closet term for fund is : ")
print(getClosestTerm("fund",tf,lsa))
print("= ="*20)
print("6 closet terms for fund is : ")
print(kClosestTerms(6,"fund",tf,lsa))

concepts = ["yield","fund","price","asset","global"]

sim_matrix = np.dot(lsa,lsa.T)
    
#print(index_array)
#get 3 closet terms for ontology concepts
kterms_f = []
for i in concepts:
    kterms = kClosestTerms(3,i,tf,lsa)
    for term in kterms:
        if term not in kterms_f:
            kterms_f.append(term)
    print(kterms)

print(kterms_f)
#append them in to ontology array
for term1 in kterms_f:
    if term1 not in concepts:
        concepts.append(term1)

print(concepts)
#get the index
index_array = []
for y in concepts:
    index = tf.vocabulary_[y]
    index_array.append(index)

print(index_array)

rows=[]
rows_f=[]
#a = np.array(a)
    
for r in index_array:
    for c in index_array:
        rows.append(sim_matrix[r][c])
        print(r," - ",c," =rows\n",rows)
    #r_array = np.array([rows])
    rows_f.append(rows)
    print("aaray\n",rows_f)
    rows=[]

rows_m = np.array(rows_f)
print("rows_m\n",rows_m)

final_sim_matrix = np.asmatrix(rows_m)
print("matirx\n",final_sim_matrix)

final_sim_matrix_df = pd.DataFrame(final_sim_matrix,columns=concepts)
print("df\n",final_sim_matrix_df)

final_sim_matrix_df.to_csv("dataframe.csv",index=False)
#r_dataframe = pandas2ri.py2ri(df)
##print(type(r_dataframe))
#print(r_dataframe)

command = 'C:/Program Files/R/R-3.5.1/bin/Rscript'
path2script = 'E:/SLIIT1/4th year/RESEARCH/Workspace/research/testr.R'
#rgs = r_dataframe
retcode = subprocess.call([command, path2script], shell=True)



os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
path = 'Igraph.dot'
s = Source.from_file(path)
s.view()