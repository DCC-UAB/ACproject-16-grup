import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import small_ratings, ground_truth
from sklearn.model_selection import train_test_split
from abc import abstractmethod

class Collaborative:
    def __init__(self, ratings=None, movies=None):
        self.ratings = ratings
        self.movies = movies
        self.ratings_matrix = None
        self.similarity = None

    @abstractmethod
    def load_data(self, ratings, movies):
        return "ERROR: no s'ha implementat la funció load_data()"

    @abstractmethod
    def calculate_similarity_matrix(self, method):
        return "ERROR: no s'ha implementat la funció calculate_similarity_matrix()"

    def predict_rating(self, user_id, movie_id, system="user",topN=5, similarity_threshold=0.1):
        """Prediu la valoració d'un usuari per una pel·lícula."""
        if system not in ["user", "item"]:
            return "ERROR: sistema assignat no disponible ('user' o 'item')"
        if user_id not in self.ratings_matrix.index or movie_id not in self.ratings_matrix.columns:
            return np.nan

        user_similarities = self.similarity.loc[user_id].drop(user_id)
        filtered_users = user_similarities[user_similarities > similarity_threshold]

        effective_topN = min(topN, len(filtered_users))
        top_similar_users = filtered_users.sort_values(ascending=False).head(effective_topN)
        ratings_for_movie = self.ratings_matrix.loc[top_similar_users.index, movie_id]
        valid_ratings = ratings_for_movie.dropna()

        if valid_ratings.empty:
            return self.ratings_matrix[movie_id].mean()

        relevant_similarities = top_similar_users[valid_ratings.index]
        numerator = (valid_ratings * relevant_similarities).sum()
        denominator = relevant_similarities.abs().sum()

        if denominator == 0:
            return self.ratings_matrix[movie_id].mean()

        predicted_rating = numerator / denominator
        if system == "user":
            return predicted_rating
        else:
            return np.clip(predicted_rating, 0, 5)

    def evaluate_model(self, data, system, topN=5):
        """Avalua el model segons MAE i RMSE."""
        predictions = data.apply(lambda row: self.predict_rating(row['user'], row['id'], system, topN=topN), axis=1)
        mae = (data['rating'] - predictions).abs().mean()
        rmse = np.sqrt(((data['rating'] - predictions) ** 2).mean())
        return mae, rmse

    @abstractmethod
    def recommend_for_user(self, user_id, topN=5):
        return "ERROR: no s'ha implementat la funció recommend_for_user()"

    def plot_errors(self, mae_val_cos, mae_val_per, rmse_val_cos, rmse_val_per):
        data = pd.DataFrame({
            'Mètode': ['Cosinus', 'Pearson', 'Cosinus', 'Pearson'],
            'Mètrica': ['MAE', 'MAE', 'RMSE', 'RMSE'],
            'Valor': [mae_val_cos, mae_val_per, rmse_val_cos, rmse_val_per]
        })

        plt.figure(figsize=(8, 6))
        sns.barplot(data=data, x='Mètrica', y='Valor', hue='Mètode', palette='pastel')
        plt.title('Comparativa d\'errors segons la similitud (conjunt de validació)')
        plt.ylabel('Valor')
        plt.xlabel('Mètrica')
        plt.legend(title='Mètode')
        plt.show()