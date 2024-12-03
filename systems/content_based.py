import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from data_preprocessing.preprocessing_csv import small_ratings, keywords

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


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
        Afegir les keywords a les dades de les pel·lícules.
        """
        key = keywords("./datasets/keywords.csv")
        self.movies = pd.merge(self.movies, key, on='id', how='left')
        self.movies= self.movies[self.movies['keywords'].apply(lambda x: len(x) > 0)]
        #no_keywords = self.movies[self.movies['keywords'].apply(lambda x: len(x) == 0)]
        self.movies = pd.merge(self.movies, self.credits, on='id', how='left')


 






if __name__ == '__main__':
    recomender = ContentBasedRecomender()
    recomender.merge_data()
    print(recomender.movies.columns)
    user_id = 1 
