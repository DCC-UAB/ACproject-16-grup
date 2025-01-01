import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from sklearn.model_selection import train_test_split
from data_preprocessing.preprocessing_csv import small_ratings, ground_truth
from systems.collaborative import Collaborative

class ItemItemRecommender(Collaborative):
    def __init__(self, ratings=None, items=None):
        super().__init__(ratings, items)

    def load_data(self):
        """Carrega i processa les dades."""
        self.ratings_matrix = self.ratings.pivot_table(index='id', columns='user', values='rating')

    def calculate_similarity_matrix(self, method='cosine'):
        """Calcula la matriu de similitud entre ítems."""
        if method == 'cosine':
            similarity = cosine_similarity(self.ratings_matrix.fillna(0))
            self.similarity = pd.DataFrame(similarity, index=self.ratings_matrix.index, columns=self.ratings_matrix.index)
        elif method == 'pearson':
            similarity = self.ratings_matrix.corr(method='pearson')
            self.similarity = similarity.reindex(index=self.ratings_matrix.index, columns=self.ratings_matrix.index)
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson'.")

    def recommend_for_user(self, user_id, topN=5):
        """Recomana ítems a un usuari basant-se en els ítems més valorats per aquest usuari."""
        if user_id not in self.ratings_matrix.columns:
            print(f"Warning: Usuari {user_id} no trobat a les dades.")
            return pd.DataFrame(columns=['title', 'predicted_rating'])

        user_ratings = self.ratings_matrix[user_id]
        mean_user_rating = user_ratings.mean()  # Normalitzem les valoracions
        rated_items = user_ratings.dropna().sort_values(ascending=False).index

        recommendations = {}
        for item_id in rated_items:
            similar_items = self.similarity.loc[item_id].drop(item_id).sort_values(ascending=False)
            for similar_item, similarity in similar_items.items():
                if similar_item not in user_ratings or pd.isna(user_ratings[similar_item]):
                    if similarity > 0.1:  # Aplicar umbral de similitud
                        if similar_item not in recommendations:
                            recommendations[similar_item] = 0
                        # Utilitzem la valoració normalitzada
                        recommendations[similar_item] += similarity * (user_ratings[item_id] - mean_user_rating)

        # Normalitzem per la suma de similituds
        for item in recommendations:
            recommendations[item] = np.clip(recommendations[item] + mean_user_rating, 0, 5)  # Limitar el rango a [0, 5]

        recommended_items = pd.Series(recommendations).sort_values(ascending=False).head(topN)
        recommended_items = recommended_items.reset_index()
        recommended_items.columns = ['id', 'predicted_rating']
        recommended_items = pd.merge(recommended_items, self.items[['id', 'title']], on='id', how='left')

        return recommended_items[['title', 'predicted_rating']]

if __name__ == "__main__":
    start_time = time.time()
    # Exemple d'ús
    recommender = ItemItemRecommender()

    # Carregar les dades
    ratings, movies, _, _ = small_ratings()
    ground_truth_df, ratings = ground_truth(ratings)
    train_data, test_data = train_test_split(ratings, train_size=0.7, random_state=42)
    recommender.load_data(train_data, movies)

    user_id = 100

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
    recommender.calculate_similarity_matrix(method='pearson')
    recommendations = recommender.recommend_for_user(user_id, topN=5)
    print(f"Recomanacions per a l'usuari {user_id}:")
    print(recommendations)

    end_time = time.time()  # Temps al final de l'execució
    print(f"Temps total d'execució: {end_time - start_time:.2f} segons")

    # Plot errors
    recommender.plot_errors(mae_val_cos, mae_val_per, rmse_val_cos, rmse_val_per)
