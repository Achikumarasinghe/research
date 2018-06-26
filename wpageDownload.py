#! /usr/bin/python
#
# crawler.py  (version of 2015-10-09)
#
# Recursively download web pages, and store them to files
#
# Usage:
#
#	python crawler.py article_name
#
# e.g.,
#
#	python crawler.py Business_intelligence
#
# NOTE:
# This code is provided for illustration purposes, and is not suitable for large-scale
# document retrieval.
#
# CHANGELOG:
#
# 2015-10-09
# - Added a 1-second pause between downloads
# - Added the FOLLOWLOCATION option to follow redirects
# - Changed filenames to avoid the '/' problem in name
#   (e.g., Wikipedia's "Spin 1/2" article), interpreted as a subfolder
#   by the filesystem
#
# 2015-09-23
# - Removed document preprocessing:
#     moved "preprocess" function to process.py
# - Saving documents to folder "pages"
#
# 2015-09-16
# - Original version

import sys
import pycurl
from StringIO import StringIO
import re
import time

# Download a single web page via the pycurl package
# and return its text
def download (url):
	buffer = StringIO()
	c = pycurl.Curl()
	c.setopt(c.URL, url)
	c.setopt(c.WRITEFUNCTION, buffer.write)
	c.setopt(c.FOLLOWLOCATION, True)
	c.perform()
	c.close()
	return buffer.getvalue()

# Iterates all anchors contained in the string "body" whose href starts by
# "/wiki/" and does not contain colons.
# Every time such a URL is found, yield it. For example, if the following anchor is found:
#   <a href="/wiki/Business">
# then the function yields the string "Business"
href = re.compile(r'''href\s*=\s*"/wiki/([^":]+)"''')
def extract_urls (body):
	for link in href.findall(body):
		yield link

# Get the Wikipedia article name from the command line
base_url = sys.argv[1]

# Initially, the download queue only contains the base URL
queue = [base_url]
# No visited pages at the beginning
visited = []

# Maximum number of pages to download
pages_to_download = 10

# Repeat until the maximum is reached
for i in range(pages_to_download):
	# Take the next article to download
	url = queue[0]
	# Remove it from the queue
	del queue[0]
	print ('Downloading ' + url + '...')
	# The URL is relative (just the article name), append the remaining part and download
	body = download('https://en.wikipedia.org/wiki/'+url)
	# Save the preprocessed body on a file with the same name as the article
	f = open ('pages/'+url.replace('/','|'), 'w')
	f.write (body)
	f.close ()
	# Add the downloaded page to the list of already visited link
	visited.append (url)
	# For every URL in the downloaded page
	for link in extract_urls(body):
		# If it is the case, add the URL to the queue for future retrieval.
		if link not in visited and link not in queue:
			queue.append (link)
	# Relax for 1 second to avoid annoying the server
	time.sleep (1)