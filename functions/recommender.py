import pickle

movies = None
similarity = None

def load_model():
    global movies, similarity
    with open('model/movie_list.pkl', 'rb') as f:
        movies = pickle.load(f)
    with open('model/similarity.pkl', 'rb') as f:
        similarity = pickle.load(f)

def recommend(movie_title):
    if movie_title not in movies['title'].values:
        return [], []

    index = movies[movies['title'] == movie_title].index[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    recommended_titles = []
    recommended_ids = []

    for i in distances:
        movie = movies.iloc[i[0]]
        recommended_titles.append(movie.title)
        recommended_ids.append(movie.id)

    return recommended_titles, recommended_ids

load_model()
