import os
import joblib
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data if not already downloaded
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

nltk.download('stopwords', download_dir=nltk_data_dir)
nltk.download('wordnet', download_dir=nltk_data_dir)

# Initialize stop words and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    # Lemmatize
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Create directory for models if it doesn't exist
model_dir = 'models'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Define categories
categories = [
    "Hair and Grooming", "Body Firming & Lifting Treatments", "Body Sculpting",
    "Body Skin Care", "Braids, Locs & Twists", "Day Spa", "Dental", "Driving Class",
    "Eyebrows", "Eyelash Customisation", "Face Skin Care", "Fitness & Sport",
    "Hair Care & Texture Services", "Hair Colouring", "Hair Extensions", "Hair Removal",
    "Hair Transplant", "Haircuts & Styling", "Holistic Medicine", "Home Improvement & Maintenance",
    "Home Services", "Makeup", "Massage", "Mechanics", "Nails", "Nutritionist",
    "Permanent Makeup", "Pet Services", "Photography", "Physical Therapy", "Piercing",
    "Podiatry", "Removal of Skin Imperfections", "Scar & Mole Removal", "Tattoo", "Tutor",
    "Yoga", "Creative Design", "Animation", "Art and Illustration", "Dental Services",
    "Event Services Event Planning", "Fashion and Styling", "Graphic Design",
    "Interior Design", "Music and Audio", "Marketing and Advertising", "Skin Care",
    "Teaching and Coaching", "Videography", "Writing and Editing"
]

# Preprocess categories
categories = [preprocess_text(cat) for cat in categories]

# Load a lighter-weight Sentence-BERT model
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# Function to preprocess and embed text using the Sentence-BERT model
def embed_text(texts):
    return model.encode(texts)

# Compute or load category embeddings
embeddings_path = os.path.join(model_dir, 'category_embeddings.pkl')
if os.path.exists(embeddings_path):
    category_vectors = joblib.load(embeddings_path)
else:
    category_vectors = embed_text(categories)
    joblib.dump(category_vectors, embeddings_path)

@app.route('/get_suggestions', methods=['POST'])
def get_suggestions():
    try:
        data = request.json
        job_title = data.get('title')
        if not job_title:
            return jsonify({"error": "Title is required"}), 400

        # Preprocess job title
        job_title_processed = preprocess_text(job_title)
        # Embed the job title using the Sentence-BERT model
        job_vector = embed_text([job_title_processed])

        # Calculate similarities
        similarities = cosine_similarity(job_vector, category_vectors).flatten()

        # Get top 5 categories
        top_indices = similarities.argsort()[-5:][::-1]
        top_categories = [categories[i] for i in top_indices]

        return jsonify(top_categories)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)