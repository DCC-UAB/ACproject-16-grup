import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from user_to_user import UserUserRecommender
from item_to_item import ItemItemRecommender
from content_based import ContentBasedRecomender
from data_preprocessing.preprocessing_csv import small_ratings
from svd import SVDRecommender
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from surprise import SVD
from surprise import Dataset, Reader, accuracy


user_id = int(input("Introdueix l'ID de l'usuari: "))
rates, movies = small_ratings()
movie_id = int(input("Introdueix l'ID de la pel·lícula: "))

# User-User
user_user = UserUserRecommender()
user_user.load_data('./datasets/ratings_small.csv')
predicted_rating_cosine = user_user.predict_rating(user_id, movie_id, topN=20, method='cosine')
print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")
predicted_rating_pearson = user_user.predict_rating(user_id, movie_id, topN=20, method='pearson')
print(f"Predicció de valoració (pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")

# Item-Item
item_item = ItemItemRecommender()
rec=item_item.recommend_for_user(user_id, similarity='Cosine',top_n=5)
print('Recomanacions cosinus:\n',rec)

#Content-Based
movie = input("Introdueix el títol de la pel·lícula: ")
content_based = ContentBasedRecomender()
content_based.merge_data()
tfidf_matrix = content_based.tfidf()
#Coficient de Pearson
pearson_sim = np.corrcoef(tfidf_matrix.toarray())
r=content_based.find_similar(movie, 10, pearson_sim)
print('Recomanacions Pearson:\n',r)

#Distància cosinus
cosine_sim = linear_kernel(tfidf_matrix)
r=content_based.find_similar(movie, 10, cosine_sim)
print(r,'\n')

#SVD
svd = SVDRecommender()
reader = Reader(rating_scale=(0.5, 5)) 
data = Dataset.load_from_df(rates[['userId', 'movieId', 'rating']], reader)
#Train model
model = SVD(n_factors=50, random_state=42) 
predictions = svd.train_model()
mse = accuracy.mse(predictions)
print(f"Root Mean Squared Error (MSE): {mse:.4f}")
#Recommendations
recomendations=svd.recommend_for_user(10, 5)
print("Top recomanacions:")
for movie_id, score in recomendations:
    print(f"Pel·lícula ID: {movie_id}, Predicció: {score:.2f}")


