from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from dotenv import load_dotenv
import os
from groq import Groq
from scipy.sparse import csr_matrix

load_dotenv()

app = Flask(__name__)

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Load data
ratings = pd.read_csv('/Users/Aman/Desktop/Mogen/ml-100k/u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
movies = pd.read_csv('/Users/Aman/Desktop/Mogen/ml-100k/u.item', sep='|', encoding='ISO-8859-1', names=['movie_id', 'title'], usecols=[0, 1])
ratings = ratings.merge(movies, on='movie_id')

user_movie_matrix = ratings.pivot_table(index='user_id', columns='title', values='rating', aggfunc='mean').fillna(0)
user_movie_matrix_sparse = csr_matrix(user_movie_matrix.values)
U, sigma, Vt = svds(user_movie_matrix_sparse, k=50)
sigma = np.diag(sigma)
predicted_ratings = np.dot(np.dot(U, sigma), Vt)
predicted_ratings_df = pd.DataFrame(predicted_ratings, columns=user_movie_matrix.columns)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    feedback = data.get('feedback')
    recommendations = recommend_movies(user_id, ratings, predicted_ratings_df, feedback=feedback)
    return jsonify({"recommendations": recommendations})

def recommend_movies(user_id, original_ratings, predicted_ratings_df, feedback=None, num_recommendations=5):
    user_ratings = original_ratings[original_ratings['user_id'] == user_id]
    already_watched = user_ratings['title'].tolist()

    if feedback:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"Suggest movie genres based on this feedback: {feedback}"}
            ],
            model="llama3-8b-8192",
            stream=False,
        )
        genres = chat_completion.choices[0].message.content.split(", ")
        filtered_movies = [title for title in predicted_ratings_df.columns if any(genre.lower() in title.lower() for genre in genres)]
        user_predicted_ratings = predicted_ratings_df.loc[user_id - 1, filtered_movies].sort_values(ascending=False)
    else:
        user_predicted_ratings = predicted_ratings_df.iloc[user_id - 1].sort_values(ascending=False)

    recommended_movies = [movie for movie in user_predicted_ratings.index if movie not in already_watched]
    return recommended_movies[:num_recommendations]

if __name__ == '__main__':
    app.run(debug=True)