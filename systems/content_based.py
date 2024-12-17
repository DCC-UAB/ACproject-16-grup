import sys
import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Ajusta el sistema de directorios para importar correctamente
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import small_ratings, keywords
class ContentBasedRecommender:
    def __init__(self):
        self.ratings, self.movies = small_ratings()
        self.credits = pd.read_csv("./datasets/credits.csv")
        self.similarity_matrix = None
        self.similarity_method = None

    def merge_data(self):
        key = keywords("./datasets/keywords.csv")
        self.movies = pd.merge(self.movies, key, on='id', how='left')
        self.movies = pd.merge(self.movies, self.credits, on='id', how='left')
        return self.movies

    def compute_similarity(self, method='cosine'):
        self.similarity_method = method
        tfidf = TfidfVectorizer(stop_words='english')
        no_key = self.movies[self.movies['keywords'] == '']
        self.movies = self.movies[self.movies['keywords'].notna()]
        self.movies['keywords'] = self.movies['keywords'].apply(lambda x: ' '.join(x))
        self.movies = self.movies[self.movies['keywords'] != '']
        tfidf_matrix=tfidf.fit_transform(self.movies["keywords"])

        if method == 'cosine':
            self.similarity_matrix = linear_kernel(tfidf_matrix)
        elif method == 'pearson':
            tfidf_array = tfidf_matrix.toarray()
            self.similarity_matrix = np.corrcoef(tfidf_array)
        else:
            raise ValueError("Invalid similarity method. Choose 'cosine' or 'pearson'.")

    def find_similar_for_user(self, user_id, n):
        user_movies = self.ratings[self.ratings['user'] == user_id]
        top_rated_movies = user_movies.sort_values(by='rating', ascending=False).head(5)
        watched_movie_ids = top_rated_movies['id'].unique()

        all_movie_ids = self.movies['id'].values
        not_watched_ids = set(all_movie_ids) - set(watched_movie_ids)
        watched_indices = self.movies[self.movies['id'].isin(watched_movie_ids)].index

        recommendations = {}
        for idx in watched_indices:
            sim_scores = list(enumerate(self.similarity_matrix[idx]))
            for movie_idx, score in sim_scores:
                movie_id = self.movies.iloc[movie_idx]['id']
                if movie_id in not_watched_ids:
                    recommendations[movie_id] = max(recommendations.get(movie_id, 0), score)

        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[1:n+1]
        return pd.DataFrame(
            [(self.movies.loc[self.movies['id'] == movie_id, 'title'].values[0], similarity) 
             for movie_id, similarity in sorted_recommendations],
            columns=['Title', 'Similarity']
        )

if __name__ == '__main__':
    recommender = ContentBasedRecommender()
    recommender.merge_data()
    similarity_method = 'cosine' #Pearson
    recommender.compute_similarity(method=similarity_method)

    user_id = 1
    recommendations = recommender.find_similar_for_user(user_id, 10)
    print(f"Recomenacions per l'usuari {user_id} amb similaritat {similarity_method}:")
    print(recommendations)