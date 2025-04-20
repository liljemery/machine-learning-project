import requests
import os
from dotenv import load_dotenv

# Cargar variables del archivo .env
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(url)
    data = response.json()
    poster_path = data.get("poster_path")
    if not poster_path:
        return "https://via.placeholder.com/500x750?text=No+Image"
    return f"https://image.tmdb.org/t/p/w500/{poster_path}"