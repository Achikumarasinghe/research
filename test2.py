#! /usr/bin/python
#
# documents.py  (version of 2015-11-04)
#
# Reads all preprocessed documents in the "processed_pages" folder,
# transforms them into sets and into term frequency vectors.
# Computes their similarity wrt a query. Applies SVD.
#
# Usage:
#
#	python documents.py
#
# Needs a populated "processed_pages" folder.
#
# NOTE:
# This code is provided for illustration purposes, and is not suitable for large-scale
# document retrieval.
#
# CHANGELOG:
#
# 2015-11-04:
#
# - Use of SVD to identify relevant "topics" in the document set
#
# 2015-10-28:
# - Singular vector decomposition
# - reduced-rank approximation of cosine similarity
#
# 2015-10-21
# - Top-10 listings for various metrics
# - MinHash for Jaccard coefficient estimate
#
# 2015-10-07
# - Merged the two files into documents.py
#
# 2015-09-23
# - Original version of TF.py and jaccard.py

import sys
import glob
import numpy
from matplotlib import pyplot

######################################
#
# Document input

# Structures holding documents and terms
document_list = []	# will contain a list of documents, stored as dictionaries.
					# index in this list is used as ID of document
document_ids = {}	# reverse search: from document name to corresponding ID
token_list = []		# List of token strings; index is token's ID
token_ids = {}		# Reverse dictionary: from token string to ID

# Repeat for every document in processed_pages
for filename in glob.glob('data/*'):
	# Read the document as list of blank-separated tokens
	f = open (filename, 'r')
	tokens = f.read().split()
	f.close()
	# Get the document name as last part of path
	article_name = filename[filename.rfind('/')+1:]
	sys.stderr.write ('Processing document %s...\n' % article_name)
	# Document's ID is the length of the current document list
	doc_id = len(document_list)
	# Insert ID in inverse list
	document_ids[article_name] = doc_id
	# Populate token structure for all tokens in document
	for t in tokens:
		# Only if token hasn't been seen yet
		if t not in token_ids:
			# Token's ID is token list length
			token_ids[t] = len(token_list)
			# Append token to list
			token_list.append(t)
	# Transform the document's token list into the corresponding ID list
	tids = [token_ids[t] for t in tokens]
	# Store the document as both its token ID list and the corresponding set
	# Also remember the document's name
	document_list.append({
		'name': article_name,
		'tokens': tids,
		'set': set(tids)
	})

# At the end of the loop, we have the total number of documents and tokens
number_of_documents = len(document_list)
number_of_tokens = len(token_list)
sys.stderr.write ('%d documents, %d tokens\n' % (number_of_documents, number_of_tokens))

##############################################
#
# Building the TF-IDF matrix

sys.stderr.write ('Building the TF matrix and counting term occurrencies\n')
# For each term, count how many documents contain it (to compute IDF)
token_count = [0] * number_of_tokens
# Alloc the |T|x|D| TFIDF matrix. No need to initialize its entries
TFIDF = numpy.empty((number_of_tokens,number_of_documents), dtype=float)
# Scan the document list
for i,doc in enumerate(document_list):
	# For each term, count the number of occurrences within the document
	# Initialize with zeros
	n_dt = [0] * number_of_tokens
	# For all token IDs in document
	for tid in doc['tokens']:
		# if first occurrence, increase global count for IDF
		if n_dt[tid] == 0:
			token_count[tid] += 1
		# increase local count
		n_dt[tid] += 1
	# Normalize local count by document length obtaining TF vector;
	# store it as the i-th column of the TFIDF matrix.
	TFIDF[:,i] = numpy.array(n_dt, dtype=float) / len(doc['tokens'])

# Transform the global count into IDF
sys.stderr.write ('Computing the IDF vector\n')
IDF = numpy.log10(number_of_documents / numpy.array(token_count, dtype=float))

# Apply IDF multipliers to the rows of the TF matrix (left-multiply by diagonal IDF values)
sys.stderr.write ('Multiplying IDF coefficients into the TF matrix...\n')
# First method: explicitly multiply each row by the appropriate IDF coefficient
for row,coeff in zip(TFIDF,IDF):
	row *= coeff
#
# Second method: beware, numpy.diag(IDF)
# is a VERY large matrix, and should be avoided if the dictionary contains
# more than a few tokens
#TFIDF = numpy.diag(IDF).dot(TFIDF)

###############################################
#
# Cosine similarity to a query

sys.stderr.write ('Computing query similarities...\n')

# Assume a query text (already normalized)
query = 'MONEY MARKET TRADING STOCKS INVESTMENTS BULL COMPANY INDUSTRY VALUE PRICES'

# Split the query into its blank-separated tokens
q_split = query.split()
q_tokens = set()

# Compute the occurrences of each token in the query
q_count = [0] * number_of_tokens
q_length = 0
for token in q_split:
	try:
		# Try updating the token's count by finding its ID
		t_id = token_ids[token]
		q_count[t_id] += 1
		q_length += 1
		q_tokens.add(t_id)
	except:
		# If a query token doesn't have an ID, discard it
		pass
# Normalize by query length and multiply (elementwise) by IDF
# to get TF-IDF representation of query
q_TFIDF = (numpy.array(q_count,dtype=float) / q_length) * IDF
print("q_tfidf")
print(q_TFIDF)

# In order to compute cosine similarity, we need to normalize each
# document (TFIDF column) by its length (2-norm)
TFIDF_norm = TFIDF.dot(numpy.diag([1/numpy.linalg.norm(col) for col in TFIDF.T]))
# Same normalization applies to the query vector
q_TFIDF_norm = q_TFIDF / numpy.linalg.norm(q_TFIDF)

# Array of cosine similarities (dot products of normalized document and query vectors)
# is obtained by matrix-vector multiplication.
sim = TFIDF_norm.T.dot(q_TFIDF_norm)

# Function to print the names of the "N" documents with the highest
# ranking in the list "sim"; if "sim" contains distances (i.e., the smaller the better),
# then set "smallest" to True.
def print_top(sim,N,smallest=False):
	sorted_sim = sorted(enumerate(sim),key=lambda t:t[1], reverse=not smallest)
	for i,s in sorted_sim[:N]:
		print(i, s, document_list[i]['name'])

# Jaccard coefficient: size of intersection / size of union
def jaccard (A, B):
	return float(len(A&B)) / float(len(A|B))

# Array of Jaccard similarities between the set representation of the query
# and the representations of the documents
jsim = [jaccard(q_tokens,doc['set']) for doc in document_list]

# Euclidean distance between two vectors
def distance(v1, v2):
	return numpy.linalg.norm(v1-v2)

# Array of Euclidean distances between the (non normalized) TFIDF
# representation of the query and the representations of the documents
dist = [distance(q_TFIDF,d) for d in TFIDF.T]

# Print the three top-10 listings
print("Top cosine similarities:")
print_top (sim,10)
print ("Top Jaccard similarities:")
print_top (jsim,10)
print ("Top Euclidean distances:")
print_top (dist,10,smallest=True)

###########################################



######################################################
#
# SVD for rank reduction

# Compute the Singular Value Decomposition of the normalized TFIDF matrix
print ("Computing the SVD...")
U, Sigma, VT = numpy.linalg.svd (TFIDF_norm, full_matrices=0)
print (TFIDF_norm.shape, U.shape, Sigma.shape, VT.shape)
print (Sigma)

# Compute the cosine similarity array given the SVD decomposition of
# the TFIDF matrix (computed above), the normalized TFIDF query vector q
# and the desired rank r1
def reduced_similarity (r1, U, Sigma, VT, q):
	U_r1 = U[:,:r1]
	Sigma_r1 = Sigma[:r1]
	VT_r1 = VT[:r1]
	q_r1 = numpy.diag(Sigma_r1).dot(U_r1.T).dot(q_TFIDF_norm)
	return VT_r1.T.dot(q_r1)

# Test the above function
r1 = 100
sim_r1 = reduced_similarity (r1, U, Sigma, VT, q_TFIDF_norm)

print ("Top cosine similarities after SVD:")
print_top (sim_r1,10)

# Plot the true similarities vs. the reduced-rank ones
pyplot.plot(sim,sim_r1,'.')
pyplot.title ('Comparison of true vs. reduced-rank cosine similarities')
pyplot.xlabel('Cosine similarity')
pyplot.ylabel('Reduced-rank approximation (r1=%d)' % r1)
pyplot.show()

# Repeat the test for all values of r1
print ("Testing reduced-rank similarities for all ranks...")
diff = []
r1_range = range(1,len(Sigma)+1)
sim_norm = numpy.linalg.norm(sim)
for r1 in r1_range:
	sys.stderr.write ('\r%d' % r1)
	sim_r1 = reduced_similarity (r1, U, Sigma, VT, q_TFIDF_norm)
	diff.append (numpy.linalg.norm(sim-sim_r1)/sim_norm)
# Plot the results
pyplot.plot(r1_range,diff)
pyplot.title ('Relative error of reduced-rank approximation')
pyplot.xlabel('Reduced rank r1')
pyplot.ylabel('Relative RMSE')
pyplot.show()

# For the 10 top-ranked SVD dimensions (with the highest singular vectors):
# print the top-10 (largest positive SVD coefficient) and bottom-10 (largest negative SVD coefficient)
# tokens and document titles.
# This lets us identify "topics" in our document set, in particular from dimension 1
print ("Top-10 topics in SVD:")
for column in range(10):
	print ('Topic %d (%f)' % (column, Sigma[column]))
	relevance = sorted(enumerate(U[:,column]),key=lambda t:t[1],reverse=True)
	for i in range(10):
		print ('T %-15s %6.3f %-15s %6.3f' % (
			token_list[relevance[i][0]], relevance[i][1],
			token_list[relevance[-i-1][0]], relevance[-i-1][1]))
	relevance = sorted(enumerate(VT[column]),key=lambda t:t[1],reverse=True)
	for i in range(10):
		print ('D %-25s %6.3f %-25s %6.3f' % (
			document_list[relevance[i][0]]['name'], relevance[i][1],
			document_list[relevance[-i-1][0]]['name'], relevance[-i-1][1]))