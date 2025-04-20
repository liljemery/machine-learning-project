"""
generate_model.py

Este script:
- Lee el dataset de películas
- Limpia y procesa los textos
- Calcula la matriz de similitud
- Guarda todo como archivos .pkl
"""

import pandas as pd
import re
import pickle
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

movies_df = pd.read_csv("data/top10K-TMDB-movies.csv")
movies = movies_df[['id', 'title', 'overview', 'genre']].copy()

movies['overview'] = movies['overview'].fillna('')
movies['genre'] = movies['genre'].fillna('')

movies['tags'] = movies['overview'] + " " + movies['genre']

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

movies['tags_clean'] = movies['tags'].apply(clean_text)
new_data = movies[['id', 'title', 'tags_clean']]

cv = CountVectorizer(max_features=10000, stop_words='english')
vector = cv.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()

similarity = cosine_similarity(vector)

with open('model/movie_list.pkl', 'wb') as f:
    pickle.dump(new_data, f)

with open('model/similarity.pkl', 'wb') as f:
    pickle.dump(similarity, f)

print("✅ Modelo generado y guardado exitosamente.")
