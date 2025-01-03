import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import time
import sys
import os
import pandas as pd
import sys
import os

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import small_ratings, ground_truth
from sklearn.model_selection import train_test_split
from systems.collaborative import Collaborative

class UserUserRecommender(Collaborative):
    def __init__(self, ratings=None, movies=None):
        super().__init__(ratings, movies)

    def load_data(self):
        """Carrega i processa les dades."""
        self.ratings_matrix = self.ratings.pivot_table(index='user', columns='id', values='rating')

    def calculate_similarity_matrix(self, method='cosine'):
        """Calcula la matriu de similitud entre usuaris."""
        if method == 'cosine':
            similarity = cosine_similarity(self.ratings_matrix.fillna(0))  # Omplim NaN temporalment amb 0
        elif method == 'pearson':
            similarity = self.ratings_matrix.T.corr(method='pearson').values
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson'.")
        self.similarity = pd.DataFrame(similarity, index=self.ratings_matrix.index, columns=self.ratings_matrix.index)

    def recommend_for_user(self, user_id, topN=5):
        """Recomana pel·lícules a un usuari."""
        user_ratings = self.ratings_matrix.loc[user_id]
        not_watched = user_ratings[user_ratings.isnull()].index

        predictions = {}
        for movie_id in not_watched:
            predictions[movie_id] = self.predict_rating(user_id, movie_id, system="user", topN=topN)

        recommended = pd.Series(predictions).sort_values(ascending=False)
        recommended=recommended[:topN]
        recommended_movies = recommended.reset_index()
        recommended_movies.columns = ['id', 'predicted_rating']
        recommended_movies = pd.merge(recommended_movies, self.movies[['id', 'title']], on='id', how='left')

        return recommended_movies[['title', 'predicted_rating']]

if __name__ == "__main__":
    start_time = time.time()

    # Carregar les dades
    ratings, movies, _, _ = small_ratings()
    train_data, test_data = train_test_split(ratings, train_size=0.7, random_state=42)
    ground_truth_df, ratings = ground_truth(ratings)

    recommender = UserUserRecommender(train_data, movies)
    recommender.load_data()

    # Avaluació amb similitud cosinus
    recommender.calculate_similarity_matrix(method='cosine')
    mae_val_cos, rmse_val_cos = recommender.evaluate_model(ground_truth_df, system="user")
    print(f"Validació (Cosinus) - MAE: {mae_val_cos:.4f}, RMSE: {rmse_val_cos:.4f}")

    # Avaluació amb correlació de Pearson
    recommender.calculate_similarity_matrix(method='pearson')
    mae_val_per, rmse_val_per = recommender.evaluate_model(ground_truth_df, system="user")
    print(f"Validació (Pearson) - MAE: {mae_val_per:.4f}, RMSE: {rmse_val_per:.4f}")

    # Avaluació amb conjunt de test
    mae_test, rmse_test = recommender.evaluate_model(test_data, system="user")
    print(f"Test - MAE: {mae_test:.4f}, RMSE: {rmse_test:.4f}")

    # Recomanacions per a un usuari
    user_id = 133
    recommendations = recommender.recommend_for_user(user_id, topN=5)
    print(f"Recomanacions per a l'usuari {user_id}:\n{recommendations}")

    print(f"Temps total: {time.time() - start_time:.2f} segons")

    # Plot errors
    recommender.plot_errors(mae_val_cos, mae_val_per, rmse_val_cos, rmse_val_per)
