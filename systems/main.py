import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from user_to_user import UserUserRecommender
from item_to_item import ItemItemRecommender
from content_based import ContentBasedRecommender
from data_preprocessing.preprocessing_csv import small_ratings
from svd import SVDRecommender
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD
from surprise import Dataset, Reader, accuracy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score
import pandas as pd


user_id = int(input("Introdueix l'ID de l'usuari: "))
rates, movies, keywords, credits = small_ratings()
movie_id = int(input("Introdueix l'ID de la pel·lícula: "))

# User-User
print("\nUser-User")

user_user = UserUserRecommender()
user_user.load_data(rates, movies)
user_user.calculate_similarity_matrix(method='cosine')
predicted_rating_cosine = user_user.predict_rating(user_id, movie_id, topN=5, similarity_threshold=0.1)
print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")
user_user.calculate_similarity_matrix(method='pearson')
predicted_rating_pearson = user_user.predict_rating(user_id, movie_id, topN=5, similarity_threshold=0.1)
print(f"Predicció de valoració (pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")

actual_rating = rates[(rates['user'] == user_id) & (rates['id'] == movie_id)]['rating'].values
if actual_rating.size > 0:
    actual_rating = actual_rating[0]
    print(f"Valoració real per l'usuari {user_id} per la pel·lícula {movie_id}: {actual_rating}")
    print(f"Error (cosinus): {abs(predicted_rating_cosine - actual_rating):.4f}")
    print(f"Error (pearson): {abs(predicted_rating_pearson - actual_rating):.4f}")

    mse_cosine = mean_squared_error([actual_rating], [predicted_rating_cosine])
    print(f"Mean Squared Error (cosinus): {mse_cosine:.4f}")

    mse_pearson = mean_squared_error([actual_rating], [predicted_rating_pearson])
    print(f"Mean Squared Error (pearson): {mse_pearson:.4f}")

    print(f"Root Mean Squared Error (cosinus): {np.sqrt(mse_cosine):.4f}")
    print(f"Root Mean Squared Error (pearson): {np.sqrt(mse_pearson):.4f}")
else:
    print(f"No hi ha valoració real per l'usuari {user_id} per la pel·lícula {movie_id}")


# Item-Item
print("\nItem-Item")
item_item = ItemItemRecommender()
item_item.load_data('./datasets/ratings_small.csv')
rec = item_item.recommend_for_user(user_id, top_n=5, method='cosine')
print('Recomanacions cosinus:\n',rec)

# Content-Based
print("\nContent-Based")
recommender = ContentBasedRecommender()
recommender.merge_data()
# Pearson
similarity_method = 'pearson'
recommender.compute_similarity(method=similarity_method)
recommendations = recommender.find_similar_for_user(user_id, 10, 3)
print(f"Recomenacions per l'usuari {user_id} amb similaritat {similarity_method}:")
print(recommendations)
# Cosine
similarity_method = 'cosine'
recommender.compute_similarity(method=similarity_method)
recommendations = recommender.find_similar_for_user(user_id, 10, 3)
print(f"Recomenacions per l'usuari {user_id} amb similaritat {similarity_method}:")
print(recommendations)

# SVD
print("\nSVD")
rates = pd.read_csv("./datasets/ratings_small.csv")
model = SVD(n_factors=50, random_state=42) 
reader = Reader(rating_scale=(0.5, 5)) 
data = Dataset.load_from_df(rates[['userId', 'movieId', 'rating']], reader)
svd = SVDRecommender(model, rates, data)
predictions = svd.train_model()
recomendations=svd.recommend_for_user(10, 5)
print("Top recomanacions:")
for movie_id, score in recomendations:
    print(f"Pel·lícula ID: {movie_id}, Predicció: {score:.2f}")