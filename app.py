import streamlit as st
import pandas as pd
import requests
import ast
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables (API Key)
load_dotenv()

# --- 1. DATA LOADING & CLEANING ---
@st.cache_data
def load_data():
    # Load and merge datasets
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    
    # Keep only necessary columns
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average']]
    df.dropna(inplace=True)

    # Helper functions to clean JSON-like strings
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]
    
    def convert_cast(obj):
        return [i['name'] for i in ast.literal_eval(obj)][:3] # Top 3 actors

    def fetch_director(obj):
        return [i['name'] for i in ast.literal_eval(obj) if i['job'] == 'Director']

    # Apply cleaning
    df['genres'] = df['genres'].apply(convert)
    df['keywords'] = df['keywords'].apply(convert)
    df['cast'] = df['cast'].apply(convert_cast)
    df['crew'] = df['crew'].apply(fetch_director)
    df['overview'] = df['overview'].apply(lambda x: x.split())

    # Collapse words (remove spaces between names like 'Johnny Depp' -> 'JohnnyDepp')
    def collapse(L):
        return [i.replace(" ","") for i in L]

    df['cast'] = df['cast'].apply(collapse)
    df['crew'] = df['crew'].apply(collapse)
    df['genres'] = df['genres'].apply(collapse)
    df['keywords'] = df['keywords'].apply(collapse)

    # Create the tags / Content Soup
    df['tags'] = df['genres'] + df['keywords'] + df['cast'] + df['crew'] + df['overview']
    df['tags'] = df['tags'].apply(lambda x: " ".join(x))
    
    return df

df = load_data()

# --- 2. THE ALGORITHMIC CORE ---
cv = CountVectorizer(stop_words='english')
vector = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vector)

# --- 3. POSTER FETCHING FUNCTION ---
def fetch_poster(movie_id):
    api_key = os.getenv("TMDB_API_KEY")
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}"
    try:
        data = requests.get(url).json()
        poster_path = data['poster_path']
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Movie Match AI", layout="wide")
st.title("🎬 Movie Recommender System")
st.markdown("Find your next favorite movie using **Cosine Similarity**.")

selected_movie = st.selectbox("Type or select a movie:", df['title'].values)

if st.button('Recommend'):
    idx = df[df['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]
    
    cols = st.columns(5)
    for i, col in enumerate(cols):
        movie_idx = distances[i][0]
        movie_id = df.iloc[movie_idx].movie_id
        title = df.iloc[movie_idx].title
        rating = df.iloc[movie_idx].vote_average
        
        with col:
            st.image(fetch_poster(movie_id))
            st.write(f"**{title}**")
            st.caption(f"⭐ Rating: {rating}")
