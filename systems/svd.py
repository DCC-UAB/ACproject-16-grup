from surprise import SVD
from surprise import Dataset, Reader, accuracy
from surprise.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

class SVDRecommender:
    def __init__(self, model, ratings, data):
        self.model = model
        self.ratings = ratings
        self.data = data
        self.train, self.test = train_test_split(data, test_size=0.2, random_state=42)

    def train_model(self):
        self.model.fit(self.train)
        predictions = self.model.test(self.test)
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
    
    #Llista per n_factors
    n_factors_list = [10, 20, 50, 100, 150, 200]
    rmse_values = []

    #Bucle per provar diferents n_factors
    for n in n_factors_list:
        model = SVD(n_factors=n, random_state=42)
        svd = SVDRecommender(model, ratings, data)

        predictions = svd.train_model()
        rmse = accuracy.rmse(predictions, verbose=False)
        rmse_values.append(rmse)
        print(f"n_factors: {n}, RMSE: {rmse:.4f}")

    #Gràfica
    plt.figure(figsize=(8, 6))
    plt.plot(n_factors_list, rmse_values, marker='o', linestyle='-', color='r')
    plt.title("n_factor que s'ajusta més", fontsize = 14)
    plt.xlabel("n_factors", fontsize = 12)
    plt.ylabel("RMSE",fontsize = 12)
    plt.grid(True)
    plt.show

    #Més Baix: n_factor = 10
    #Baix, però agafa relacions més complexes: n_factor = 50
    '''
    -> Escull n_factors=10 si busques un model més senzill, 
    amb un rendiment lleugerament millor (menor RMSE).

    -> Escull n_factors=50 si el lleuger augment en el RMSE no és una preocupació 
    i vols explorar un model més expressiu i potent per capturar relacions més complexes.
    '''
    model = SVD(n_factors=50, random_state=42) 
    svd=SVDRecommender(model, ratings, data)

    predictions = svd.train_model()
    rmse = accuracy.rmse(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


    recomendations=svd.recommend_for_user(10, 5)
    print("Top recomanacions:")
    for movie_id, score in recomendations:
        print(f"Pel·lícula ID: {movie_id}, Predicció: {score:.2f}")
