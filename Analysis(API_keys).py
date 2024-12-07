import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from groq import Groq

# Initialize Groq client
client = Groq()

# Load and prepare data
ratings = pd.read_csv('/Users/Aman/Desktop/u.data', sep='\t', 
                      names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('/Users/Aman/Desktop/ml-100k/u.item', sep='|', 
                     encoding='ISO-8859-1', names=['movie_id', 'title'], usecols=[0, 1])
ratings = ratings.merge(movies, on='movie_id')

# Create user-movie matrix and decompose with SVD
user_movie_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating', aggfunc='mean').fillna(0)
U, sigma, Vt = svds(user_movie_matrix, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_movie_matrix.columns)

def analyze_preferences(feedback):
    """
    Use Groq API to analyze user feedback and get genre preferences.
    """
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"Suggest movie genres based on this feedback: {feedback}",
            }
        ],
        model="llama3-8b-8192",
        stream=False,
    )
    
    genres = chat_completion.choices[0].message.content
    print("AI Suggested Genres:", genres)
    return genres

def recommend_movies(user_id, original_ratings, predicted_ratings_df, num_recommendations=5, feedback=None):
    """
    Recommend movies for a user based on their feedback and predicted ratings.
    """
    user_ratings = original_ratings[original_ratings['user_id'] == user_id]
    already_watched = user_ratings['title'].tolist()
    
    # If feedback is provided, analyze it and filter recommendations by genre
    if feedback:
        genres = analyze_preferences(feedback)
        # Filter movies by genre (assuming we have a genre-movie mapping)
        # simulating it by matching genres to movie titles
        # (needs genre data for each movie)
        genre_keywords = genres.split(", ")
        filtered_movies = [title for title in predicted_ratings_df.columns 
                           if any(genre.lower() in title.lower() for genre in genre_keywords)]
        user_predicted_ratings = predicted_ratings_df.loc[user_id - 1, filtered_movies].sort_values(ascending=False)
    else:
        user_predicted_ratings = predicted_ratings_df.iloc[user_id - 1].sort_values(ascending=False)

    # Recommend movies that the user hasn't watched yet
    recommended_movies = [movie for movie in user_predicted_ratings.index if movie not in already_watched]
    return recommended_movies[:num_recommendations]