from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error
import numpy as np

import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from data_preprocessing.preprocessing_csv import small_ratings, ground_truth

class SVDRecommender:
    def __init__(self, ratings=None, movies=None, data=None, n_factors=50, random_state=42):
        self.model = self.model_n_factors(n_factors)
        self.movies = movies
        self.ratings = ratings
        self.data = data
        self.train, self.test = train_test_split(data, test_size=0.2, random_state=random_state)

    def model_n_factors(self, n_factors):
        return SVD(n_factors, random_state=42)

    def train_model(self, model=None):
        model = model if model is not None else self.model

        model.fit(self.train)
        predictions = model.test(self.test)
        return predictions

    def recommend_for_user(self, user_id, top_n=5):
        self.train_model()
        user_ratings = self.ratings[self.ratings['user'] == user_id]['id'].values
        all_movie_ids = self.ratings['id'].unique()
        unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_ratings]
        predicted_scores = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
        predicted_scores.sort(key=lambda x: x[1], reverse=True)
        predicted_scores = pd.DataFrame(predicted_scores, columns=['id', 'score'])
        predicted_scores = pd.merge(predicted_scores, self.movies[['id', 'title']], on='id', how='left')
        return predicted_scores[:top_n]

    def get_true_and_predicted(self):
        true_ratings = []
        predicted_ratings = []
        for uid, iid, true_r, est, _ in self.model.test(self.test):
            true_ratings.append(true_r)
            predicted_ratings.append(est)
        return np.array(true_ratings), np.array(predicted_ratings)

    def get_root_mean_squared_error(self, n_factors_list):
        rmse_train_values = []
        rmse_test_values = []

        for n in n_factors_list:
            model = self.model_n_factors(n)
            predictions = self.train_model(model)
            rmse_train = accuracy.rmse(predictions, verbose=False)

            test_predictions = model.test(self.test)
            rmse_test = accuracy.rmse(test_predictions, verbose=False)

            rmse_train_values.append(rmse_train)
            rmse_test_values.append(rmse_test)

            print(f"n_factors: {n}, RMSE Train: {rmse_train:.4f}, RMSE Test: {rmse_test:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(n_factors_list, rmse_train_values, marker='o', linestyle='-', color='r', label='Train RMSE')
        plt.plot(n_factors_list, rmse_test_values, marker='o', linestyle='-', color='b', label='Test RMSE')
        plt.title("RMSE segons n_factors")
        plt.xlabel("n_factors")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()

    def get_model_metrics(self):
        predictions = self.train_model()
        rmse = accuracy.rmse(predictions, verbose=False)

        # Separem els ratings que són True amb els predits pel model
        true_ratings, predicted_ratings = self.get_true_and_predicted()
        mae = mean_absolute_error(true_ratings, predicted_ratings)

        # Diem que els ratings superiors a la meitat són positius -> "threshold"
        true_labels = (true_ratings > 3).astype(int)
        predicted_scores = (predicted_ratings > 3).astype(int)

        # Calculem les diferents mètriques
        precision = precision_score(true_labels, predicted_scores)
        recall = recall_score(true_labels, predicted_scores)
        f1 = f1_score(true_labels, predicted_scores)
        accuracy_metric = accuracy_score(true_labels, predicted_scores)
        self.plot_precision_recall_curve(true_labels, predicted_ratings)
        return precision, recall, f1, accuracy_metric, rmse, mae

    def plot_precision_recall_curve(self, true_labels, predicted_ratings):
        precision_values, recall_values, _ = precision_recall_curve(true_labels, predicted_ratings)

        false_positive_rate, true_positive_rate, _ = roc_curve(true_labels, predicted_ratings)
        roc_area = auc(false_positive_rate, true_positive_rate)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(recall_values, precision_values, color='blue', label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(false_positive_rate, true_positive_rate, color='blue', label=f'ROC curve (AUC = {roc_area:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    ratings, movies, _, _ = small_ratings()
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings[['user', 'id', 'rating']], reader)
    svd = SVDRecommender(ratings,movies, data)
    n_factors_list = [10, 20, 50, 100, 150, 200]
    svd.get_root_mean_squared_error(n_factors_list)

    recomendations = svd.recommend_for_user(10, 5)
    print(recomendations)

    precision, recall, f1, acc, rsme, mae = svd.get_model_metrics()
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"RMSE: {rsme:.4f}")
    print(f"MAE: {mae:.4f}")