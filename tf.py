#! /usr/bin/python
#
# TF.py  (version of 2015-09-23)
#
# Reads all preprocessed documents in the "processed_pages" folder,
# transforms them into term frequency vectors and computes all pairwise
# Euclidean distances and cosine similarities.
#
# Usage:
#
#	python TF.py
#
# Needs a populated "processed pages" folder.
#
# NOTE:
# This code is provided for illustration purposes, and is not suitable for large-scale
# document retrieval.
#
# CHANGELOG:
#
# 2015-09-23
# - Original version

import glob
import numpy

# Compute the Euclidean distance (2-norm of difference)
# between vectors stored as numpy arrays
def distance (d1, d2):
	return numpy.linalg.norm(d1-d2)

# Compute the cosine similarity (normalized dot product)
# between vectors stored as numpy arrays
def cosine_similarity (d1, d2):
	return d1.dot(d2) / (numpy.linalg.norm(d1)*numpy.linalg.norm(d2))

# Maintain a dictionary that associates the document name to its representation
documents = {}

# Span all preprocessed documents
for filename in glob.glob('processed_pages/*'):
	# Import the document as a list of blank-separated tokens
	f = open (filename, 'r')
	tokens = f.read().split()
	f.close()
	# Get the article name from the filename
	article_name = filename[filename.rfind('/')+1:]
	# Store the token list
	documents[article_name] = tokens
	print ("Article %s has %d tokens." % (
		article_name, len(documents[article_name]))

# After all documents have been collected into memory,
# build a list of all distinct tokens and associate them to a numeric ID
#
# Direct correspondence: ID to token
token_list = []
#
# Inverse correspondence: token to ID
token_ids = {}
#
# Span all token lists
for tokens in documents.values():
	# For all tokens in a document
	for t in tokens:
		# If the token hasn't been found yet
		if t not in token_ids:
			# use the current list lehgth as token ID
			token_ids[t] = len(token_list)
			token_list.append(t)
#
# Finally, get the number of distinct tokens
number_of_tokens = len(token_list)

# Now convert all document token lists to Term Frequency vectors
for article_name,tokens in documents.items():
	# Count all occurrences of each token in the document
	count = [0] * number_of_tokens
	for t in tokens:
		count[token_ids[t]] += 1
	# Normalize the counts by the total document length and
	# store the resulting numpy array alongside the
	# original token list in the dictionary
	l = float(len(tokens))
	TF = [c/l for c in count]
	documents[article_name] = (tokens, numpy.array(TF))
#
# At the end of this pass, the "documents" dictionary associates every article name
# with a 2-uple containing the token list and the TF array

#For all pairs of documents, print their Euclidean distance and their cosine similarities
for k1 in documents:
	for k2 in documents:
		d1 = documents[k1][1]
		d2 = documents[k2][1]
		dist = distance (d1, d2)
		sim = cosine_similarity (d1, d2)
		print "%35s %35s %f %f" % (k1, k2, dist, sim)
# By plotting the output of this nested loop, it is possible to visually observe
# a negative correlation between the two numbers.