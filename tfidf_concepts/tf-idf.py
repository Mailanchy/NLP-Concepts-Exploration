# tfidf vectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#1. Setup a sample data corpus.
documents = [
    "I love coding in Python.",
    "Python is a great tool for Data Science.",
    "I love Data Science.",
    "The sun is shining today.", 
    "The cat sat on the mat."
]

#2. Vectorize: Initialize and fit the TF-IDF model
vectorizer = TfidfVectorizer(stop_words='english') #created the obj
# stop_words='english' is the only available parameter as of i know and it removes every kind of words like [the,is,a,i,in,]
#Its an optional parameter
tfidf_matrix = vectorizer.fit_transform(documents) # returns a sparse amtrix

feature_names = vectorizer.get_feature_names_out() #returns all unique words in the corpus
df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names) #for readability
# .toarray for converting the sparse matrix to normal
df.index = [f"Doc {i+1}" for i in range(len(documents))] # Name the rows

print("\n--- TF-IDF Vector Space Representation (Scores close to 1 are important) ---")
print(df.round(3))
