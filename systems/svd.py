from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
class SVDRecommender:
    def __init__(self, model, ratings, data):
        self.model = model
        self.ratings = ratings
        self.data = data
        self.train, self.test = train_test_split(data, test_size=0.2, random_state=42)

        
    def train_model(self):
        self.model.fit(self.train)
        predictions = model.test(self.test)
        return predictions
    
    def recommend_for_user(self, user_id, top_n=5):
        user_ratings = self.ratings[self.ratings['userId'] == user_id]['movieId'].values
        all_movie_ids = self.ratings['movieId'].unique()
        unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_ratings]
        predicted_scores = [(movie_id, self.model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
        predicted_scores.sort(key=lambda x: x[1], reverse=True)
        return predicted_scores[:top_n]
        

if __name__ == '__main__':
    ratings = pd.read_csv("./datasets/ratings_small.csv")
    reader = Reader(rating_scale=(0.5, 5)) 
    data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    model = SVD(n_factors=50, random_state=42) 
    svd=SVDRecommender(model, ratings, data)

    predictions = svd.train_model()
    mse = accuracy.mse(predictions)
    print(f"Root Mean Squared Error (MSE): {mse:.4f}")


    recomendations=svd.recommend_for_user(10, 5)
    print("Top recomanacions:")
    for movie_id, score in recomendations:
        print(f"Pel·lícula ID: {movie_id}, Predicció: {score:.2f}")
