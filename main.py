import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from difflib import get_close_matches
import ast

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")
netflix = pd.read_csv("netflix_titles.csv")
movies = movies.merge(credits, on='title')
movies = movies[['movie_id', 'title', 'overview', 'genres', 'cast']]
movies.dropna(inplace=True)
netflix = netflix[['title', 'description', 'listed_in', 'cast']]
netflix.rename(columns={'description': 'overview', 'listed_in': 'genres'}, inplace=True)
netflix['movie_id'] = range(
    movies['movie_id'].max() + 1,
    movies['movie_id'].max() + 1 + len(netflix)
)
netflix.fillna('', inplace=True)
netflix = netflix[['movie_id', 'title', 'overview', 'genres', 'cast']]
movies = pd.concat([movies, netflix], ignore_index=True)
movies.drop_duplicates(subset='title', inplace=True)
movies.fillna('', inplace=True)
def convert(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)]
    except:
        return [i.strip().replace(" ", "") for i in obj.split(',')]
def convert3(obj):
    try:
        return [i['name'] for i in ast.literal_eval(obj)[:3]]
    except:
        return [i.strip().replace(" ", "") for i in obj.split(',')[:3]]
movies['genres'] = movies['genres'].apply(convert)
movies['cast'] = movies['cast'].apply(convert3)
movies['overview'] = movies['overview'].astype(str).apply(lambda x: x.split())
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['tags'] = movies['overview'] + movies['genres'] + movies['cast']
movies = movies[['movie_id', 'title', 'tags']]
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x)).str.lower()
movies.loc[movies['title'].str.contains("indian|hindi|bollywood", case=False, na=False), 'tags'] += " bollywood india hindi"
tfidf = TfidfVectorizer(max_features=8000, stop_words='english')
vectors = tfidf.fit_transform(movies['tags']).toarray()
similarity = cosine_similarity(vectors)
def recommend(movie):
    movie = movie.lower()
    titles = movies['title'].str.lower().tolist()
    match = get_close_matches(movie, titles, n=1, cutoff=0.6)
    if not match:
        print("Movie not found in database.")
        return
    movie = match[0]
    idx = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[idx]
    results = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:8]
    print("\nRecommended Movies:\n")
    for i in results:
        print("-", movies.iloc[i[0]].title)
while True:
    movie_name = input("\nEnter movie name (or type exit): ")
    if movie_name.lower() == "exit":
        break
    recommend(movie_name)
