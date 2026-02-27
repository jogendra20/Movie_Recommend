import streamlit as st
import pandas as pd
import requests
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. DATA PROCESSING ---
@st.cache_data
def load_and_process_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    df = movies.merge(credits, on='title')
    
    # Select and clean features
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew', 'vote_average', 'release_date']]
    df.dropna(inplace=True)
    
    def convert(obj):
        return [i['name'] for i in ast.literal_eval(obj)]

    df['genres'] = df['genres'].apply(convert)
    df['keywords'] = df['keywords'].apply(convert)
    df['cast'] = df['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
    df['crew'] = df['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job']=='Director'])
    
    # Remove spaces and combine
    df['tags'] = df['genres'] + df['keywords'] + df['cast'] + df['crew']
    df['tags'] = df['tags'].apply(lambda x: " ".join([i.replace(" ","") for i in x]))
    return df

df = load_and_process_data()

# --- 2. THE AI MODEL ---
cv = CountVectorizer(stop_words='english')
vector = cv.fit_transform(df['tags']).toarray()
similarity = cosine_similarity(vector)

# --- 3. FETCH POSTER FROM API ---
def fetch_poster(movie_id):
    # USE YOUR API KEY HERE
    api_key = "YOUR_TMDB_API_KEY"
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={a9e7cb6061a49231b9648a395c6261e5}"
    try:
        response = requests.get(url).json()
        return "https://image.tmdb.org/t/p/w500/" + response['poster_path']
    except:
        return "https://via.placeholder.com/500x750?text=No+Poster+Found"

# --- 4. STREAMLIT UI ---
st.set_page_config(page_title="Joe's Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender AI")

selected_movie = st.selectbox("Select a movie you like:", df['title'].values)

if st.button('Recommend'):
    idx = df[df['title'] == selected_movie].index[0]
    distances = sorted(list(enumerate(similarity[idx])), reverse=True, key=lambda x: x[1])[1:6]
    
    cols = st.columns(5)
    for i, col in enumerate(cols):
        movie_id = df.iloc[distances[i][0]].movie_id
        title = df.iloc[distances[i][0]].title
        rating = df.iloc[distances[i][0]].vote_average
        
        with col:
            st.image(fetch_poster(movie_id))
            st.write(f"**{title}**")
            st.caption(f"⭐ Rating: {rating}")
