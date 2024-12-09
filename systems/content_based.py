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
        """
        Inicializa el sistema de recomendación con los datos de valoraciones y películas.
        """
        self.ratings, self.movies = small_ratings()
        self.credits = pd.read_csv("./datasets/credits.csv")
        self.item_similarity = None
        self.cast_similarity = None
        self.keywords_similarity = None

    def merge_data(self):
        """
        Merge: Añade las keywords, el cast y el crew a los datos de las películas.
        """
        key = keywords("./datasets/keywords.csv")
        self.movies = pd.merge(self.movies, key, on='id', how='left')
        
        movies_with_keywords = self.movies[self.movies['keywords'].apply(lambda x: len(x) > 0)]
        movies_without_keywords = self.movies[self.movies['keywords'].apply(lambda x: len(x) == 0)]
        
        self.movies = pd.merge(self.movies, self.credits, on='id', how='left')
        return movies_with_keywords, movies_without_keywords


    def tfidf(self):
        """
        Crea una matriz TF-IDF para el contenido de las películas basado en la columna 'overview'.
        """
        tfidf = TfidfVectorizer(stop_words='english')
        self.movies['overview'] = self.movies['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.movies["overview"])
        return tfidf_matrix

    def find_similar(self, title, n, sim):
        """
        Función para recomendar utilizando una métrica de similitud.
        """
        index = pd.Series(self.movies.index, index=self.movies.title).drop_duplicates()
        idx = index[title]
        sim_score = list(enumerate(sim[idx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
        recommended_idx = [i[0] for i in sim_score][1:n+1]
        return pd.DataFrame(self.movies['title'].iloc[recommended_idx])


if __name__ == '__main__':
    recommender = ContentBasedRecommender()    
    movies_with_keywords, movies_without_keywords = recommender.merge_data()
    print(movies_with_keywords["overview"].head())
    tfidf_matrix = recommender.tfidf()

    # Similitud amb Pearson
    tfidf_array = tfidf_matrix.toarray()
    pearson_sim = np.corrcoef(tfidf_array)
    print("Recomendaciones con Pearson:")
    recommendations_pearson = recommender.find_similar("Casino", 10, pearson_sim)
    print(recommendations_pearson, '\n')

    # Similitud amb coseno
    cosine_sim = linear_kernel(tfidf_matrix)
    print("Recomendaciones con Coseno:")
    recommendations_cosine = recommender.find_similar("Casino", 10, cosine_sim)
    print(recommendations_cosine, '\n')
