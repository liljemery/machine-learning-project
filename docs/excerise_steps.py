"""
    In this project, I will be creating a movie recommendation system 
    using the TMDB API.

    The goal of this project is to create a movie recommendation system that 
    recommends movies to users based on their preferences and ratings.

    In the first block of code, This code is reading in the data from the TMDB API and
    storing it in a pandas dataframe "new_data", with only the columns "id", "title", "tags". 
    The "tags" column is created by combining the "overview" and "genre" columns which is then used in the next block of code.

"""
print("============Start Content from the first block of code============")
# First, we import the necessary libraries
import pandas as pd

# Then we read in the data from the TMDB CSV
movies_df = pd.read_csv("data/top10K-TMDB-movies.csv")

# Then we create a new dataframe
movies = movies_df[['id', 'title', 'overview', 'genre']].copy()

# After that, we clean the text in the "tags" column
movies['overview'] = movies['overview'].fillna('')

# Also, we clean the text in the "tags" column
movies['genre'] = movies['genre'].fillna('')

# Then, we combine the text in the "tags" column in order to use it in the next block of code as a single string
movies['tags'] = movies['overview'] + movies['genre']

# Finally, we drop the "overview" and "genre" columns since they are not needed
new_data = movies.drop(columns=['overview', 'genre'])

# Then, we print the first two rows of the dataframe to see what it looks like


print("============End Content from the first block of code============")

"""
    In the second block of code, this code is cleaning the text in the "tags" column.
    The text is cleaned by removing the punctuation, stop words, and lemmatizing the words.
    Also, the text is tokenized and joined into a single string for each movie so that it can be used in the next block of code.
"""
print("============Start Content from the second block of code============")
# In this block of code, we import the necessary libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Then, we download the necessary NLTK data for the stop words and the lemmatizer
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

# Then, we create a function to clean the text in the "tags" column
def clean_text(text):
    # First the functions validates that the input is a string.
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    
    # Then, the text does convert the text to lowercase.
    text = text.lower()

    # Then, the text does remove the punctuation.
    text = re.sub(r'[^\w\s]', '', text)

    # Then, this text is tokenized.
    words = word_tokenize(text)

    # After that, we remove the stop words.
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Then, we lemmatize the words.
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Then, we join the words into a single string.
    text = ' '.join(words)

    return text


new_data['tags_clean'] = new_data['tags'].apply(clean_text)

print("============End Content from the second block of code============")
"""

"""
print("============Start Content from the third block of code============")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

train_data, test_data = train_test_split(new_data, test_size=0.2, random_state=42)

cv = CountVectorizer(max_features=10000, stop_words='english')

vector = cv.fit_transform(new_data['tags_clean'].values.astype('U')).toarray()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)


distance = sorted(list(enumerate(similarity[2])), reverse=True, key=lambda vector: vector[1])
# for i in distance[0:6]:
#     print(new_data.iloc[i[0]].title)


def recommend(movie):
    if movie not in new_data['title'].values:
        print(f"'{movie}' not found in the dataset.")
        return

    movie_index = new_data[new_data['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda vector: vector[1])[1:6]

    for i in movies_list:
        print(new_data.iloc[i[0]].title)

recommend('Iron Man')

import pickle
import pickle

pickle.dump(new_data, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("============End Content from the third block of code============")