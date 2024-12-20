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

class ItemItemRecommender:
    def __init__(self, ratings=None, items=None):
        self.ratings = ratings
        self.items = items
        self.ratings_matrix = None
        self.item_similarity = None

    def load_data(self, ratings, items):
        """Carrega i processa les dades."""
        self.ratings = ratings
        self.items = items
        self.ratings_matrix = self.ratings.pivot_table(index='id', columns='user', values='rating')

    def calculate_similarity_matrix(self, method='cosine'):
        """Calcula la matriu de similitud entre ítems."""
        if method == 'cosine':
            similarity = cosine_similarity(self.ratings_matrix.fillna(0))
            self.item_similarity = pd.DataFrame(similarity, index=self.ratings_matrix.index, columns=self.ratings_matrix.index)
        elif method == 'pearson':
            similarity = self.ratings_matrix.corr(method='pearson')
            self.item_similarity = similarity.reindex(index=self.ratings_matrix.index, columns=self.ratings_matrix.index)
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson'.")

    def predict_rating(self, user_id, item_id, topN=5, similarity_threshold=0.1):
        """Prediu la valoració d'un usuari per un ítem."""
        if user_id not in self.ratings_matrix.columns or item_id not in self.ratings_matrix.index:
            return np.nan

        item_similarities = self.item_similarity.loc[item_id].drop(item_id)  # Excloem l'ítem actual
        filtered_items = item_similarities[item_similarities > similarity_threshold]

        effective_topN = min(topN, len(filtered_items))
        top_similar_items = filtered_items.sort_values(ascending=False).head(effective_topN)
        ratings_by_user = self.ratings_matrix.loc[top_similar_items.index, user_id]
        valid_ratings = ratings_by_user.dropna()

        if valid_ratings.empty:
            return self.ratings_matrix.loc[item_id].mean()

        relevant_similarities = top_similar_items[valid_ratings.index]
        numerator = (valid_ratings * relevant_similarities).sum()
        denominator = relevant_similarities.abs().sum()

        if denominator == 0:
            return self.ratings_matrix.loc[item_id].mean()

        predicted_rating = numerator / denominator
        return np.clip(predicted_rating, 0, 5)  # Limitar el rango a [0, 5]

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
            similar_items = self.item_similarity.loc[item_id].drop(item_id).sort_values(ascending=False)
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

    def evaluate_model(self, data, topN=5):
        """Avalua el model segons MAE i RMSE."""
        predictions = data.apply(lambda row: self.predict_rating(row['user'], row['id'], topN=topN), axis=1)
        mae = (data['rating'] - predictions).abs().mean()
        rmse = np.sqrt(((data['rating'] - predictions) ** 2).mean())
        return mae, rmse
    

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

if __name__ == "__main__":
    start_time = time.time()
    # Exemple d'ús
    recommender = ItemItemRecommender()

    # Carregar les dades
    ratings, movies, _, _ = small_ratings()
    train_data, test_data = train_test_split(ratings, train_size=0.7, random_state=42)
    ground_truth_df, ratings = ground_truth(ratings)
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
