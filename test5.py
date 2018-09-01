import nltk
from nltk.corpus import stopwords
import re
#from string import *
import string as string
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import pandas as pd
 
import numpy as np

#file = open("data/text.txt","r") 
#print(file.read())

 #for doc in file.readlines():
 #   tf = Counter()
  #  for word in doc.split():
  #      tf[word] +=1
  #  print(tf.items())


stemmer = PorterStemmer()
 
file = open("data/fin2.txt","r") 
data = file.read()
data = data.split(" ")  
 
texts=[]
for i in range(len(data)):
    texts.append(data[i])
    print(texts)
    texts[i] = texts[i].translate(string.punctuation).lower()
    texts[i] = nltk.word_tokenize(texts[i])
    texts[i] = [stemmer.stem(word) for word in texts[i] if not word in stopwords.words('english')]
    " ".join(texts[i])

print(texts)
 
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(texts)     

print(tfidf)

svd = TruncatedSVD(n_components = 2)
lsa = svd.fit_transform(tfidf.T)

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
 
def kClosestTerms(k,term,transformer,model):
 
    term = stemmer.stem(term)
    index = transformer.vocabulary_[term]
 
    model = np.dot(model,model.T)
 
    closestTerms = {}
    for i in range(len(model)):
        closestTerms[transformer.get_feature_names()[i]] = model[index][i]
 
    sortedList = sorted(closestTerms , key= lambda l : closestTerms[l])
 
    return sortedList[::-1][0:k]

print("* "*20)
print(getClosestTerm("money",transformer,lsa))