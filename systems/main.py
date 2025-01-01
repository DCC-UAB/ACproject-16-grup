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
from surprise import Dataset, Reader


def user_to_user(user_id, rates, movies, similarity='cosine', n=5):
    user_user = UserUserRecommender()
    user_user.load_data(rates, movies)
    user_user.calculate_similarity_matrix(method=similarity)
    rec = user_user.recommend_for_user(user_id, topN=n)
    print(f"Recomanacions per a l'usuari {user_id}:\n{rec}")

def item_to_item(user_id, rates, movies, similarity, n):
    item_item = ItemItemRecommender()
    item_item.load_data(rates, movies)
    item_item.calculate_similarity_matrix(method=similarity)
    rec = item_item.recommend_for_user(user_id, topN=n)
    print(rec)


def content_based(user_id, rates, movies, keywords, credits, similarity):
    content_based = ContentBasedRecommender()
    content_based.load_data(rates, movies, keywords, credits)
    content_based.merge_data()
    content_based.compute_similarity(method=similarity)
    rec = content_based.find_similar_for_user(user_id, 10, 3)
    print(rec)

def svd(user_id, rates, movies):
    reader = Reader(rating_scale=(0.5, 5)) 
    data = Dataset.load_from_df(rates[['user', 'id', 'rating']], reader)
    svd = SVDRecommender(rates, movies, data)
    svd.train_model()
    rec = svd.recommend_for_user(user_id)
    print(rec)


if __name__ == '__main__':
    user_id = int(input("Introdueix l'ID de l'usuari: "))
    rates, movies, keywords, credits = small_ratings()

    print("Recomanador User-to-User:")
    print("\n\tRecomanacions amb distància cosinus:")
    user_to_user(user_id, rates, movies, 'cosine', 5)
    print("\n\tRecomanacions amb coeficient de Pearson:")
    user_to_user(user_id, rates, movies, 'pearson', 5)
    
    print("\nRecomanador Item-to-Item:")
    print("\n\tRecomanacions amb distància cosinus:")
    item_to_item(user_id, rates, movies, 'cosine', 5)
    print("\n\tRecomanacions amb coeficient de Pearson:")
    item_to_item(user_id, rates, movies, 'pearson', 5)
   
    print("\nRecomanador Content-Based:")
    print("\n\tRecomanacions amb distància cosinus:")
    content_based(user_id, rates, movies, keywords, credits, 'cosine')
    print("\n\tRecomanacions amb coeficient de Pearson:")
    content_based(user_id, rates, movies, keywords, credits, 'pearson')
    
    print("\nRecomanador SVD:")
    svd(user_id, rates, movies)