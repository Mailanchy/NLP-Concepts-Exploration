# TF-IDF Concept Implementation


Analysis (similarity_finder.py):
----------
The higher the score (closer to 1.0), the more similar the TF-IDF vectors are.
Highly depend on the given corpus.
 eg:
 query: (doc 1) I love coding in python
 best match: (doc 3) I love data science
 [Our human mind find the doc 2 (Python is a great tool for data science) as best match]

My findings: The length of document effect the TF-IDF score of each word in that document (Especiallly TF score).
TF-IDF score for [doc 1, love] = [doc 1, python] = 0.532
Both love and python repeat for 2 times in the whole corpus.
Love in doc1 and doc3
python in doc1 and doc2
but TF-IDF score for [doc 2, python] < [doc 3, love] (0.406 < 0.577)
this change in score is due to the length of doc 2 which = 5 greater than length of doc 3 which = 3
This score difference is the underlying reason why the similarity score for doc1 and doc3 is higher than doc1 and doc2
