
# 🎬 Movie Recommendation System (TMDB + Machine Learning)

This project is a **content-based movie recommendation system** that suggests similar movies using metadata from the TMDB dataset. It includes:

- Text cleaning and lemmatization
- Cosine similarity based recommendations
- Poster fetching via TMDB API
- A web interface built with Streamlit
- Support for model retraining and custom image carousels

---

## 📁 Project Structure

```
MachineLearning/
├─ data/                 # Raw CSV dataset (TMDB movies)
│  └─ top10K-TMDB-movies.csv
├─ docs/                 # Script references
│  └─ excerise_steps.py
├─ functions/
│  ├─ generate_model.py  # Preprocess and serialize the model
│  ├─ poster_fetcher.py  # Fetches poster images from TMDB
│  └─ recommender.py     # Loads model and handles movie recommendations
├─ model/                # Serialized model files (.pkl)
│  ├─ movie_list.pkl
│  └─ similarity.pkl
├─ .env                  # API keys (e.g., TMDB)
├─ .env.example          # Template for environment variables
├─ .gitignore
├─ app.py                # Main Streamlit app (web interface)
└─ requirements.txt      # Project dependencies
```

---

## 🚀 How to Run the App

### 1. Clone the repo

```bash
git clone https://github.com/liljemery/machine-learning-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up your `.env`

Create a `.env` file in the root folder with your TMDB API key:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

You can use `.env.example` as a template.

### 4. Generate the model

```bash
python functions/generate_model.py
```

This will process the dataset and save `movie_list.pkl` and `similarity.pkl` in the `model/` folder.

### 5. Run the web app

```bash
streamlit run app.py
```

---

## 🧠 Features

- 🔍 Search and select a movie title
- 🤖 Get 5 similar movie recommendations
- 🎞 View movie posters fetched via the TMDB API
- 🌀 Regenerate model with a single click
- 🖼 Display results in a custom image carousel

---

## ✅ Example Usage

Once the app is running:

- Select **"Iron Man"** from the dropdown
- Click **"Show Recommendations"**
- You'll see:
  - 5 similar movie titles
  - Their posters in a carousel

---

## 🧰 Tech Stack

- **Python**
- **Pandas, Scikit-learn, NLTK**
- **Streamlit**
- **TMDB API**
- **dotenv**
- *(Optional)*: Custom frontend with React (image carousel)

---

## 📌 License

This project is for educational and portfolio purposes. API usage complies with TMDB's public terms of use.

---

## 🤝 Contributing

Pull requests and improvements are welcome! Just fork the project, make your changes, and submit a PR.
