import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from data_preprocessing.preprocessing_csv import small_ratings, keywords

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



class ContentBasedRecomender:
    def __init__(self):
        """
        Inicialitza el sistema de recomanació amb les dades de valoracions i articles.
        """
        self.ratings, self.movies = small_ratings()
        self.credits =pd.read_csv("./datasets/credits.csv")
        self.item_similarity = None
        self.cast_similarity = None
        self.keywords_similarity = None


    def merge_data(self):
        """
        Merge: Afegeix les keywords, el cast i el crew a les dades de les pel·lícules.
        """
        key = keywords("./datasets/keywords.csv")
        self.movies = pd.merge(self.movies, key, on='id', how='left')
        self.movies= self.movies[self.movies['keywords'].apply(lambda x: len(x) > 0)]
        #no_keywords = self.movies[self.movies['keywords'].apply(lambda x: len(x) == 0)]
        self.movies = pd.merge(self.movies, self.credits, on='id', how='left')


    def tfidf(self):
        """
        Crea una matriu TF-IDF per a un conjunt de dades.
        :param column: Nom de la columna per a la qual es vol crear la matriu.
        :return: Matriu TF-IDF
        """
        tfidf = TfidfVectorizer(stop_words='english')
        self.movies['overview'] = self.movies['overview'].fillna('')
        tfidf_matrix = tfidf.fit_transform(self.movies["overview"])
        
        return tfidf_matrix
    
    def find_similar(self, tfidf_matrix, title, n):
        """
        Funció per recomanar utilitzant el coeficient de Pearson.
        """
        index = pd.Series(self.movies.index, index=self.movies.title).drop_duplicates()
        idx = index[title]
        tfidf_array = tfidf_matrix.toarray()
        pearson_sim = np.corrcoef(tfidf_array)
        sim_score = list(enumerate(pearson_sim[idx]))
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)
        recommended_idx = [i[0] for i in sim_score][1:n+1]
        return pd.DataFrame(self.movies['title'].iloc[recommended_idx])




if __name__ == '__main__':
    recomender = ContentBasedRecomender()
    recomender.merge_data()
    tfidf_matrix = recomender.tfidf()
    r=recomender.find_similar(tfidf_matrix, "Casino", 10)
    
    print(r)