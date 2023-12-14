from flask import Flask, render_template, request
import os

import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# From NLTK, provides common stopwords like 'An, The, Who ..'
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text

# Function to calculate TF-IDF similarity
def calculate_tfidf_similarity(file1_path, file2_path):
    with open(file1_path, 'r', encoding='ISO-8859-1') as file1, open(file2_path, 'r', encoding='ISO-8859-1') as file2:
        file1_contents = file1.read()
        file2_contents = file2.read()

        file1_preprocessed = preprocess_text(file1_contents)
        file2_preprocessed = preprocess_text(file2_contents)

        # TF-IDF vectors
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([file1_preprocessed, file2_preprocessed])

        # Get the TF-IDF vectors for file1 and file2
        file1_tfidf = tfidf_matrix[0]
        file2_tfidf = tfidf_matrix[1]

        # Calculate cosine similarity
        similarity = (file1_tfidf * file2_tfidf.T).toarray()[0, 0]

        return similarity

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match_resume():
    if 'resume' not in request.files:
        return "No file uploaded"

    resume_file = request.files['resume']

    if resume_file.filename == '':
        return "No selected file"

    if resume_file:
        resume_filename = resume_file.filename
        resume_file.save(os.path.join("path/to/your/resumes_directory", resume_filename))

        resume_directory = "path/to/your/resumes_directory"
        file1_path = os.path.join(resume_directory, resume_filename)

        directory_path = "path/to/your/jobs_directory"
        max_similarity = 0.0
        most_similar_file = None
        all_files = {}

        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file2_path = os.path.join(directory_path, filename)

                similarity = calculate_tfidf_similarity(file1_path, file2_path)
                filename_without_extension = os.path.splitext(filename)[0]

                if similarity > max_similarity:
                    max_similarity = similarity
                    most_similar_file = filename_without_extension

                all_files[filename_without_extension] = similarity

        result = f"The position most suited to your resume is for a {most_similar_file} with a similarity of {max_similarity}\n\n"

        top_five = []
        all_files_sorted = sorted(all_files.items(), key=lambda x: x[1], reverse=True)

        for file_name, similarity in all_files_sorted[:5]:
            top_five.append((file_name, similarity))

        return render_template('index.html', result=result, top_five=top_five)

    return "Error processing file"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
