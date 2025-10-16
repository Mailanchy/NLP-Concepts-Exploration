from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
# 1. Vectorization
documents = [
    "I love coding in Python.",
    "Python is a great tool for Data Science.",
    "I love Data Science.",
    "The sun is shining today.", 
    "The cat sat on the mat."
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# 2. Define the query
query_index = 0
query_doc = documents[query_index]
query_vector = tfidf_matrix[query_index]

# 3. Similarity check
similarity_score = cosine_similarity(query_vector, tfidf_matrix).flatten() # cosine_similarity() returns a 2D array
sorted_indices = similarity_score.argsort()
result_index = sorted_indices[-2]
result_score = similarity_score[result_index]
result_doc = documents[result_index]

# Analysis & Output
print(f"\n--- Document Similarity Analysis ---")
print(f"CORPUS:\n{documents}\n")
print(f"QUERY DOCUMENT (Doc {query_index + 1}): {query_doc}\n\n")
print(f"Raw Similarity Scores (Doc {query_index + 1} vs. All Docs):\n{np.round(similarity_score, 3)}\n\n")
print(f"BEST MATCH:")
print(f"Document: Doc {result_index + 1}")
print(f"Sentence: {result_doc}")
print(f"Score: {result_score:.3f}")
print("---\n")

print("Analysis: \nThe higher the score (closer to 1.0), the more similar the TF-IDF vectors are.\nHighly depend on the given corpus")

