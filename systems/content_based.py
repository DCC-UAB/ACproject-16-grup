import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import small_ratings
from sklearn.feature_extraction.text import TfidfVectorizer

class ContentBasedRecommender:
    def __init__(self, ratings=None, movies=None, keywords=None, credits=None):
        self.ratings, self.movies, self.keywords, self.credits = small_ratings()
        self.similarity_matrix = None
        self.similarity_method = None

    def load_data(self, ratings, movies, keywords, credits):
        self.ratings = ratings
        self.movies = movies
        self.keywords = keywords
        self.credits = credits

    def merge_data(self):
        self.movies = pd.merge(self.movies, self.keywords, on='id', how='left')
        self.credits = self.credits[
            (self.credits['actors'].notnull() & (self.credits['actors'] != '')) |
            (self.credits['director'].notnull() & (self.credits['director'] != ''))
        ]

        self.movies = pd.merge(self.movies, self.credits, on='id', how='left')
        self.movies['keywords'] = self.movies['keywords'].fillna('').apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        self.movies['actors'] = self.movies['actors'].fillna('')
        self.movies['director'] = self.movies['director'].fillna('')

        self.movies['combined_features'] = self.movies.apply(
            lambda row: ' '.join(
                filter(None, [
                    ' '.join(row['keywords']) if isinstance(row['keywords'], list) else row['keywords'],
                    row['actors'],
                    row['director']
                ])
            ), axis=1
        )
        return self.movies


    def compute_similarity(self, method='cosine'):
        self.similarity_method = method
        tfidf = TfidfVectorizer(stop_words='english')

        self.movies['combined_features'] = self.movies['combined_features'].apply(lambda x: ' '.join(x.split()))
        self.movies = self.movies[self.movies['combined_features'] != '']
        tfidf_matrix = tfidf.fit_transform(self.movies["combined_features"])
        if method == 'cosine':
            self.similarity_matrix = linear_kernel(tfidf_matrix)
        elif method == 'pearson':
            tfidf_array = tfidf_matrix.toarray()
            self.similarity_matrix = np.corrcoef(tfidf_array)
        else:
            raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")


    def find_similar_for_user(self, user_id, n, nota_minima=3):
        user_movies = self.ratings[(self.ratings['user'] == user_id) & (self.ratings['rating'] >= nota_minima)]
        watched_movie_ids = user_movies['id'].unique()

        all_movie_ids = self.movies['id'].values
        not_watched_ids = set(all_movie_ids) - set(watched_movie_ids)
        watched_indices = self.movies[self.movies['id'].isin(watched_movie_ids)].index

        recommendations = {}
        for idx in watched_indices:
            if idx < len(self.similarity_matrix): 
                sim_scores = list(enumerate(self.similarity_matrix[idx]))
                for movie_idx, score in sim_scores:
                    movie_id = self.movies.iloc[movie_idx]['id']
                    if movie_id in not_watched_ids:
                        recommendations[movie_id] = max(recommendations.get(movie_id, 0), score)

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n+1]
        return pd.DataFrame(
            [(self.movies.loc[self.movies['id'] == movie_id, 'title'].values[0], similarity) 
            for movie_id, similarity in sorted_recommendations],
            columns=['Title', 'Similarity']
        )


    def evaluate_model(self, test_data):
        user_ratings_with_titles = pd.merge(test_data, self.movies[['id', 'title']], on='id', how='left')
        user_ratings_with_titles = user_ratings_with_titles[user_ratings_with_titles['rating'] >= 3]
        
        recommendations = []
        for user_id in user_ratings_with_titles['user'].unique():
            rated_titles = user_ratings_with_titles[user_ratings_with_titles['user'] == user_id]['title'].values
            recommended = self.find_similar_for_user(user_id, n=10)

            recommended_titles = recommended['Title'].values
            intersection = len(set(rated_titles).intersection(set(recommended_titles)))
            precision = intersection / len(recommended_titles) if len(recommended_titles) > 0 else 0
            recall = intersection / len(rated_titles) if len(rated_titles) > 0 else 0

            recommendations.append({'user_id': user_id, 'precision': precision, 'recall': recall, 'rated_titles': rated_titles, 'recommended_titles': recommended_titles})

        return pd.DataFrame(recommendations)



if __name__ == '__main__':
    recommender = ContentBasedRecommender()
    ratings, movies, keywords, credits = small_ratings()
    
    train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

    recommender.load_data(ratings, movies, keywords, credits)
    recommender.merge_data()
    recommender.compute_similarity(method="cosine")
    
    user_id = 1
    recommendations = recommender.find_similar_for_user(user_id, 10, 3)
    print(f"Recomenacions segons Cosine per l'usuari {user_id}:")
    print(recommendations)

<<<<<<< HEAD
    #train_data, test_data = train_test_split(recommender.ratings, test_size=0.2, random_state=42)
=======
    recommender.compute_similarity(method="pearson")
    recommendations = recommender.find_similar_for_user(user_id, 10, 3)
    print(f"Recomenacions segons Pearson per l'usuari {user_id}:")
    print(recommendations)

>>>>>>> 4f32a19db91c3fdabf08af2f8220937b58ee4758
    #recommender.evaluate_model(test_data)


    