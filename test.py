import sys
import os
from collections import Counter
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


#file = open("data/text.txt","r") 
#print(file.read())

 #for doc in file.readlines():
 #   tf = Counter()
  #  for word in doc.split():
  #      tf[word] +=1
  #  print(tf.items())

corpus = []
for file in glob.glob("data/*.txt"):
    with open(file, "r") as doc:
        corpus.append((file, doc.read()))

print(corpus)
tf = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

tfidf_matrix =  tf.fit_transform([content for file, content in corpus])

print(tfidf_matrix)

lsa = TruncatedSVD(n_components=2,n_iter=100)
lsa.fit(tfidf_matrix)
terms = tf.get_feature_names()

for i,comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedterms = sorted(termsInComp, key=lambda x: x[1],reverse=True)[:10]
    print("Concept %d:" % i)
    for term in sortedterms:
        print(term[0])
    print(" ")