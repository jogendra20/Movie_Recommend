# Movie Recommender AI 🎬

A content-based movie recommendation system built with **Python**, **Streamlit**, and **Scikit-Learn**. 

## 🚀 How it Works
1. **Vectorization**: Uses `CountVectorizer` to turn movie tags into numerical vectors.
2. **Cosine Similarity**: Calculates the angle between vectors to find the most similar movies.
3. **TMDB API**: Fetches real-time posters and ratings.

## 🛠️ Setup
1. Clone the repo.
2. Install requirements: `pip install -r requirements.txt`
3. Add your TMDB API Key in `app.py`.
4. Run the app: `streamlit run app.py`
