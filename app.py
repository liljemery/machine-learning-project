import streamlit as st
from functions.recommender import recommend, movies, load_model
from functions.poster_fetcher import fetch_poster
import subprocess

st.title("🎬 Movie Recommender System")

if st.button("🔄 Retrain Model"):
    with st.spinner("Regenerating model..."):
        result = subprocess.run(["python", "functions/generate_model.py"], capture_output=True, text=True)
        if result.returncode == 0:
            load_model()
            st.success("✅ Model retrained successfully!")
        else:
            st.error("❌ Failed to retrain model.")
            st.text(result.stderr)

movie_list = movies['title'].values
selected_movie = st.selectbox("Select a movie to get recommendations", movie_list)

if st.button("🎥 Show Recommendations"):
    names, ids = recommend(selected_movie)
    posters = [fetch_poster(mid) for mid in ids]

    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.text(names[i])
            st.image(posters[i])
