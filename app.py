import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read all text files in the directory
student_files = [doc for doc in os.listdir() if doc.endswith('.txt')]
student_notes = [open(file, encoding='utf-8').read() for file in student_files]

# Vectorize all documents
vectors = TfidfVectorizer().fit_transform(student_notes).toarray()

# Check pairwise similarity
plagiarism_results = []

for i in range(len(student_files)):
    for j in range(i + 1, len(student_files)):
        sim_score = cosine_similarity([vectors[i], vectors[j]])[0][1]
        plagiarism_results.append((student_files[i], student_files[j], sim_score))

# Print results
for result in plagiarism_results:
    print(result)

