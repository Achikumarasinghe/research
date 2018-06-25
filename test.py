import sys
import os
from collections import Counter
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import numpy 


document_list = []	# will contain a list of documents, stored as dictionaries.
					# index in this list is used as ID of document
document_ids = {}	# reverse search: from document name to corresponding ID
token_list = []		# List of token strings; index is token's ID
token_ids = {}		# Reverse dictionary: from token string to ID

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

number_of_tokens = len(token_list)
number_of_documents = len(document_list)
token_count = [0] * number_of_tokens
query = 'MONEY MARKET TRADING STOCKS INVESTMENTS BULL COMPANY INDUSTRY VALUE PRICES EXCHANGE CURRENCY BITCOIN MONEY'

# Split the query into its blank-separated tokens
q_split = query.split()
tf2 = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english')

tfidf_matrix2 =  tf2.fit_transform(q_split)

print(tfidf_matrix2)


