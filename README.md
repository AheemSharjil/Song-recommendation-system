# 🎧 Spotify Music Recommendation System

An industry-aligned final-year undergraduate project designed to recommend music you *actually* want to hear — using the power of machine learning, clustering, and a sprinkle of Spotify's magic (API).

---

## 🚀 Overview

In a world flooded with music, our goal was simple: **help users discover songs similar to their favorites** — fast, accurately, and intuitively. This app allows users to build a playlist, then receive smart recommendations based on the audio characteristics of the selected songs.

Built with:

- **Python**
- **Spotipy** (Spotify Web API wrapper)
- **Scikit-learn**
- **Streamlit**
- **Plotly**

---

## 📦 Features

- 🔍 **Song Search + Recommendation** — Enter a song name and year to get similar song suggestions.
- 🧠 **ML-Based Clustering** — Visualizes genre relationships using KMeans + t-SNE.
- 🖼️ **Interactive UI** — Clean and responsive interface using Streamlit.
- 🌐 **Spotify API Integration** — Retrieves live song metadata and audio features.

---

## 🛠️ Tech Stack

| Layer            | Tools Used                                      |
|------------------|-------------------------------------------------|
| Frontend         | Streamlit                                       |
| Backend Logic    | Python, Scikit-learn, NumPy, Pandas             |
| Visualization    | Plotly, t-SNE                                   |
| External API     | Spotify API (via Spotipy)                       |
| Data Sources     | `data.csv`, `data_by_genres.csv`                |

---

## 📚 How It Works

1. User adds songs (name + year) to a playlist.
2. For each song:
   - Fetch metadata from the dataset.
   - If missing, fetch via Spotify API.
3. Calculate the average feature vector.
4. Compare it with the dataset using cosine similarity.
5. Return top-N closest songs (excluding current ones).
6. For genre visualization:
   - Perform KMeans clustering on genre-level data.
   - Visualize with t-SNE in 2D scatter plot.

---

## ⚙️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/AheemSharjil/Song-recommendation-system.git
