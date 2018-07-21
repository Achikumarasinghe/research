import logging
from pprint import pprint
#logging.basicConfig( format='%(asctime)s:%(levelname)s:%(message)s', level=logging.DEBUG)
 
from gensim import corpora, models, similarities
documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",              
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
 
pprint( len( documents ))
 
stoplist = set( 'for of a the and to in'.split() )
texts = [ [word for word in document.lower().split() if word not in stoplist ] for document in documents ]
pprint( texts )
 
from collections import defaultdict
frequency = defaultdict( int )
for text in texts:
    for token in text:
        frequency[ token ] += 1
pprint( frequency )
 
texts2 = [ [ token for token in text if frequency[ token ] > 1 ] for text in texts ]
 
pprint( texts2 )
dictionary = corpora.Dictionary( texts2 )
print( dictionary )
print( dictionary.token2id )
 
newdoc = 'human computer interaction'
newvec = dictionary.doc2bow( newdoc.split() )
         
 
corpus = [ dictionary.doc2bow( text ) for text in texts ]
corpora.MmCorpus.serialize( './deerwster.mm', corpus )
corpora.SvmLightCorpus.serialize('./corpus.svmlight', corpus)
tfidf = models.TfidfModel( corpus )
corpus_tfidf = tfidf[ corpus ]
for doc in corpus_tfidf:
    print( doc )
# latent semantic analysis
lsi = models.LsiModel( corpus, id2word=dictionary, num_topics=2)
index = similarities.MatrixSimilarity( lsi[ corpus ] )
veclsi = lsi[ newvec ]
print( '- '*50 )
print( veclsi )
print( '- '*50 )

sims = index[ veclsi ]
for i, sim in enumerate( sims):
    pprint( documents[i]+":::"+newdoc+" similarity_score_is {}".format( sim )  )
 
print( "+ "*50 )
print( "= "*50 )
sims2 = index[  lsi[ dictionary.doc2bow(texts2[0])] ]
for i, sim in enumerate( sims2):
    print( documents[i]+":::"+documents[0]+" similarity_score_is {}".format( sim )  )
