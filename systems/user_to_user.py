import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import small_ratings, ground_truth
from sklearn.model_selection import train_test_split

class UserUserRecommender:
    def __init__(self, ratings=None, movies=None):
        self.ratings = ratings
        self.movies = movies
        self.ratings_matrix = None
        self.user_similarity = None

    
    def load_data(self, ratings, movies):
        """Carrega i processa les dades."""
        self.ratings = ratings
        self.movies = movies
        self.ratings_matrix = self.ratings.pivot_table(index='user', columns='id', values='rating')
    
    
    def calculate_similarity_matrix(self, method='cosine'):
        """Calcula la matriu de similitud entre usuaris."""
        if method == 'cosine':
            similarity = cosine_similarity(self.ratings_matrix.fillna(0))  # Omplim NaN temporalment amb 0
        elif method == 'pearson':
            similarity = self.ratings_matrix.T.corr(method='pearson').values
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson'.")
        self.user_similarity = pd.DataFrame(similarity, index=self.ratings_matrix.index, columns=self.ratings_matrix.index)

    
    def predict_rating(self, user_id, movie_id, topN=5, similarity_threshold=0.1):
        """Prediu la valoració d'un usuari per una pel·lícula."""
        if user_id not in self.ratings_matrix.index or movie_id not in self.ratings_matrix.columns:
            return np.nan

        user_similarities = self.user_similarity.loc[user_id].drop(user_id)
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

        return numerator / denominator

    
    def evaluate_model(self, data, topN=5):
        """Avalua el model segons MAE i RMSE."""
        predictions = data.apply(lambda row: self.predict_rating(row['user'], row['id'], topN=topN), axis=1)
        mae = (data['rating'] - predictions).abs().mean()
        rmse = np.sqrt(((data['rating'] - predictions) ** 2).mean())
        return mae, rmse
    
    
    def recomana(self, user_id, topN=5):
        """Recomana pel·lícules a un usuari."""
        user_ratings = self.ratings_matrix.loc[user_id]
        not_watched = user_ratings[user_ratings.isnull()].index

        predictions = {}
        for movie_id in not_watched:
            predictions[movie_id] = self.predict_rating(user_id, movie_id, topN=topN)

        recommended = pd.Series(predictions).sort_values(ascending=False)
        print(recommended)
        recommended=recommended[:topN]
        recommended_movies = recommended.reset_index()
        recommended_movies.columns = ['id', 'predicted_rating']
        recommended_movies = pd.merge(recommended_movies, self.movies[['id', 'title']], on='id', how='left')

        return recommended_movies[['title', 'predicted_rating']]



if __name__ == "__main__":
    start_time = time.time()
    recommender = UserUserRecommender()

    # Carregar les dades
    ratings, movies, _, _ = small_ratings()
    train_data, test_data = train_test_split(ratings, train_size=0.7, random_state=42)
    ground_truth_df, ratings = ground_truth(ratings)
    recommender.load_data(train_data, movies)

    # Avaluació amb similitud cosinus
    recommender.calculate_similarity_matrix(method='cosine')
    mae_val_cos, rmse_val_cos = recommender.evaluate_model(ground_truth_df)
    print(f"Validació (Cosinus) - MAE: {mae_val_cos:.4f}, RMSE: {rmse_val_cos:.4f}")

    # Avaluació amb correlació de Pearson
    recommender.calculate_similarity_matrix(method='pearson')
    mae_val_per, rmse_val_per = recommender.evaluate_model(ground_truth_df)
    print(f"Validació (Pearson) - MAE: {mae_val_per:.4f}, RMSE: {rmse_val_per:.4f}")

    # Avaluació amb conjunt de test
    mae_test, rmse_test = recommender.evaluate_model(test_data)
    print(f"Test - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")

    # Recomanacions per a un usuari
    user_id = 123
    recommendations = recommender.recomana(user_id, topN=5)
    print(f"Recomanacions per a l'usuari {user_id}:\n{recommendations}")

    print(f"Temps total: {time.time() - start_time:.2f} segons")

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Dades en forma de DataFrame
    import pandas as pd
    data = pd.DataFrame({
        'Mètode': ['Cosinus', 'Pearson', 'Cosinus', 'Pearson'],
        'Mètrica': ['MAE', 'MAE', 'RMSE', 'RMSE'],
        'Valor': [mae_val_cos, mae_val_per, rmse_val_cos, rmse_val_per]
    })

    # Gràfic amb Seaborn
    plt.figure(figsize=(8, 6))
    sns.barplot(data=data, x='Mètrica', y='Valor', hue='Mètode', palette='pastel')
    plt.title('Comparativa d\'errors segons la similitud (conjunt de validació)')
    plt.ylabel('Valor')
    plt.xlabel('Mètrica')
    plt.legend(title='Mètode')
    plt.show()
