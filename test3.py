#USING GENISM LIBRARY

import gensim

print(dir(gensim))

raw_documents = ["Finance is not merely about making money. It's about achieving our deep goals and protecting the fruits of our labor. It's about stewardship and, therefore, about achieving the good society.",
                 "While I encourage people to save 100 percent down for a home, a mortgage is the one debt that I don't frown upon.",
             "It is well enough that people of the nation do not understand our banking and monetary system, for if they did, I believe there would be a revolution before tomorrow morning.",
             "I believe that through knowledge and discipline, financial peace is possible for all of us.",
            "Beware of little expenses. A small leak will sink a great ship."]
print("Number of documents:",len(raw_documents))

from nltk.tokenize import word_tokenize
gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]
print(gen_docs)

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary[5])
print(dictionary.token2id['frown'])
print("Number of words in dictionary:",len(dictionary))
for i in range(len(dictionary)):
    print(i, dictionary[i])

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)
print("tfidf")
print(tf_idf)

s = 0
for i in corpus:
    s += len(i)
print(s)

sims = gensim.similarities.Similarity('/',tf_idf[corpus],
                                      num_features=len(dictionary))
print(sims)

print(type(sims))

query_doc = [w.lower() for w in word_tokenize("Wealth consists not in having great possessions, but in having few wants.")]
print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
print("query doc bow:")
print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
print("doc tfidf")
print(query_doc_tf_idf)

sims[query_doc_tf_idf]
print("similarity with docs and the query:")
print(sims[query_doc_tf_idf])