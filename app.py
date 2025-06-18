import os
import numpy as np
import pandas as pd
import plotly.express as px
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import streamlit as st
import warnings

# This MUST be the first Streamlit command
st.set_page_config(page_title="Spotify Recommender", layout="wide")

warnings.filterwarnings("ignore")

# Load datasets
@st.cache_data
def load_data():
    data = pd.read_csv("data.csv")
    genre_data = pd.read_csv('data_by_genres.csv')
    return data, genre_data

data, genre_data = load_data()

# Define utility constants
number_cols = [
    'valence', 'year', 'acousticness', 'danceability', 'duration_ms',
    'energy', 'explicit', 'instrumentalness', 'key', 'liveness',
    'loudness', 'mode', 'popularity', 'speechiness', 'tempo'
]

# Spotify credentials
sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
    client_id="64bad4f373774093b97753477b886f75",
    client_secret="5954f4a1d88941a3a8aee447ebd10dbc"
))

def find_song(name, year):
    song_data = defaultdict(list)
    results = sp.search(q=f"track:{name} year:{year}", limit=1)
    if not results['tracks']['items']:
        return None
    track = results['tracks']['items'][0]
    audio_features = sp.audio_features(track['id'])[0]
    song_data['name'] = name
    song_data['year'] = year
    song_data['explicit'] = int(track['explicit'])
    song_data['duration_ms'] = track['duration_ms']
    song_data['popularity'] = track['popularity']
    for key, value in audio_features.items():
        song_data[key] = value
    return pd.DataFrame(song_data, index=[0])

def get_song_data(song, spotify_data):
    filtered_songs = spotify_data[
        (spotify_data['name'] == song['name']) &
        (spotify_data['year'] == song['year'])
    ]
    if not filtered_songs.empty:
        return filtered_songs.iloc[0]
    else:
        return find_song(song['name'], song['year'])

def get_mean_vector(song_list, spotify_data):
    song_vectors = []
    for song in song_list:
        song_data = get_song_data(song, spotify_data)
        if song_data is None:
            print(f"Warning: {song['name']} does not exist in Spotify or in the database.")
            continue
        song_vector = song_data[number_cols].values
        song_vectors.append(song_vector)
    return np.mean(song_vectors, axis=0)

def flatten_dict_list(dict_list):
    flattened_dict = defaultdict(list)
    for dictionary in dict_list:
        for key, value in dictionary.items():
            flattened_dict[key].append(value)
    return flattened_dict

def recommend_songs(song_list, spotify_data, n_songs=10):
    metadata_cols = ['name', 'year', 'artists']
    song_dict = flatten_dict_list(song_list)
    song_center = get_mean_vector(song_list, spotify_data)
    scaler = StandardScaler().fit(spotify_data[number_cols])
    scaled_data = scaler.transform(spotify_data[number_cols])
    scaled_song_center = scaler.transform(song_center.reshape(1, -1))
    distances = cdist(scaled_song_center, scaled_data, metric='cosine')
    indices = np.argsort(distances.flatten())[:n_songs]
    recommended_songs = spotify_data.iloc[indices]
    return recommended_songs[~recommended_songs['name'].isin(song_dict['name'])][metadata_cols].to_dict('records')

def perform_clustering(genre_data):
    cluster_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('kmeans', KMeans(n_clusters=10, n_init=10))
    ])
    X = genre_data.select_dtypes('number')
    cluster_pipeline.fit(X)
    genre_data['cluster'] = cluster_pipeline.predict(X)

    tsne_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('tsne', TSNE(n_components=2, verbose=0, random_state=42))
    ])
    genre_embedding = tsne_pipeline.fit_transform(X)
    projection = pd.DataFrame(data=genre_embedding, columns=['x', 'y'])
    projection['genres'] = genre_data['genres']
    projection['cluster'] = genre_data['cluster']
    return projection

# Streamlit App UI
st.title("Spotify Music Recommendation System")

# Sidebar
with st.sidebar:
    st.title("🎵 Music Recommender")
    st.markdown("Discover songs similar to your favorites")
    
    # Song input
    song_name = st.text_input("Song Name", placeholder="e.g. Bohemian Rhapsody")
    song_year = st.number_input("Year", min_value=1900, max_value=2023, value=2020)
    
    # Playlist management
    if st.button("Add to Playlist"):
        if 'playlist' not in st.session_state:
            st.session_state.playlist = []
        st.session_state.playlist.append({'name': song_name, 'year': song_year})
    
    # Display current playlist
    if 'playlist' in st.session_state and st.session_state.playlist:
        st.subheader("Your Playlist")
        for i, song in enumerate(st.session_state.playlist):
            st.write(f"{i+1}. {song['name']} ({song['year']})")
    
    # Recommendation controls
    num_recs = st.slider("Number of Recommendations", 5, 20, 10)
    
    if st.button("Get Recommendations") and 'playlist' in st.session_state:
        with st.spinner("Finding similar songs..."):
            st.session_state.recommendations = recommend_songs(
                st.session_state.playlist, 
                data, 
                n_songs=num_recs
            )

# Main Content
tab1, tab2 = st.tabs(["Recommendations", "Genre Analysis"])

with tab1:
    if 'recommendations' in st.session_state and st.session_state.recommendations:
        st.subheader("Recommended Songs")
        cols = st.columns(3)
        for i, song in enumerate(st.session_state.recommendations):
            with cols[i % 3]:
                st.markdown(f"**{song['name']}**")
                st.caption(f"Artist: {song['artists']}  \nYear: {song['year']}")
    elif 'playlist' not in st.session_state:
        st.info("Add songs to your playlist to get recommendations")
    else:
        st.warning("No recommendations found. Try different songs.")

with tab2:
    st.subheader("Genre Clusters")
    if st.button("Analyze Genres"):
        with st.spinner("Processing genre clusters..."):
            projection = perform_clustering(genre_data)
            fig = px.scatter(
                projection, x='x', y='y', color='cluster', 
                hover_data=['genres'], title="Genre Clusters (t-SNE)"
            )
            st.plotly_chart(fig, use_container_width=True)