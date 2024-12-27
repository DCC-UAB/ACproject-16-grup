from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from data_preprocessing.preprocessing_csv import small_ratings, ground_truth
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_absolute_error
class SVDRecommender:
    def __init__(self, ratings=None, movies=None, data=None, n_factors=50, random_state=42, train=None, test=None):
        self.model = self.model_n_factors(n_factors)
        self.movies = movies
        self.ratings = ratings
        self.data = data
        if test is None or train is None:
            self.train, self.test = train_test_split(data, test_size=0.2, random_state=random_state)
        else:
            self.train = train
            self.test = test

    def model_n_factors(self, n_factors):
        return SVD(n_factors, random_state=42)

    def train_model(self, model=None):
        model = model if model is not None else self.model
        model.fit(self.train)

    def evaluate_model(self, model, data):
        predictions = model.test(data)
        rmse = accuracy.rmse(predictions, verbose=False)
        mae = accuracy.mae(predictions, verbose=False)
        return rmse, mae

    def get_true_and_predicted(self):
        true_ratings = []
        predicted_ratings = []
        for uid, iid, true_r, est, _ in self.model.test(self.test):
            true_ratings.append(true_r)
            predicted_ratings.append(est)
        return np.array(true_ratings), np.array(predicted_ratings)
    
    def get_model_metrics(self):
        predictions = self.model.test(self.test)
        rmse = accuracy.rmse(predictions, verbose=False)
        true_ratings, predicted_ratings = self.get_true_and_predicted()
        mae = mean_absolute_error(true_ratings, predicted_ratings)
        return rmse, mae

    def get_root_mean_squared_error(self, n_factors_list):
        rmse_train_values = []
        rmse_test_values = []

        for n in n_factors_list:
            model = self.model_n_factors(n)
            model.fit(self.train)

            trainset_test = self.train.build_testset() 
            train_rmse, _ = self.evaluate_model(model, trainset_test)
            rmse_train_values.append(train_rmse)

            test_rmse, _ = self.evaluate_model(model, self.test)
            rmse_test_values.append(test_rmse)

        return rmse_train_values, rmse_test_values

    def plot_rmse(self, n_factors_list, rmse_train_values, rmse_test_values):
        plt.figure(figsize=(10, 6))
        plt.plot(n_factors_list, rmse_train_values, marker='o', linestyle='-', label='Train RMSE')
        plt.plot(n_factors_list, rmse_test_values, marker='o', linestyle='-', label='Test RMSE')
        plt.title("RMSE for Different n_factors")
        plt.xlabel("n_factors")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.show()


    def recommend_for_user(self, user_id, top_n=5):
        user_ratings = self.ratings[self.ratings['user'] == user_id]['id'].values
        all_movie_ids = self.ratings['id'].unique()
        unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_ratings]

        predicted_scores = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
        predicted_scores.sort(key=lambda x: x[1], reverse=True)

        recommendations = pd.DataFrame(predicted_scores[:top_n], columns=['id', 'predicted_score'])
        if self.movies is not None:
            recommendations = recommendations.merge(self.movies[['id', 'title']], on='id', how='left')

        return recommendations

if __name__ == '__main__':
    ratings, movies, _, _ = small_ratings()
    ground_truth_df, ratings = ground_truth(ratings)
    reader = Reader(rating_scale=(0.5, 5))

    data = Dataset.load_from_df(ratings[['user', 'id', 'rating']], reader)
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    svd = SVDRecommender(ratings, movies, data, train=train, test=test)
    svd.train_model()
    rsme, mae = svd.get_model_metrics()
    print(f"RMSE: {rsme:.4f}")
    print(f"MAE: {mae:.4f}")

    n_factors_list = [10, 20, 50, 100, 150, 200]
    rmse_train_values, rmse_test_values = svd.get_root_mean_squared_error(n_factors_list)

    svd.plot_rmse(n_factors_list, rmse_train_values, rmse_test_values)
    print("Per n_factors:", n_factors_list)
    print("RMSE al conjunt de train:", rmse_train_values)
    print("RMSE al conjunt de test:", rmse_test_values)

    user_id = 1
    rec = svd.recommend_for_user(user_id)
    print(f"Recomanacions per a l'usuari {user_id}:\n{rec}")



