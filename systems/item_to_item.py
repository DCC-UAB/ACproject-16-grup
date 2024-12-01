import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import movies_metadata, ratings

class ItemItemRecommender:
    def __init__(self):
        """
        Inicialitza el sistema de recomanació amb les dades de valoracions i articles.
        """
        self.ratings = ratings('./datasets/ratings_small.csv')
        self.items = movies_metadata('./datasets/movies_metadata.csv')
        self.item_similarity = None
    
    # def read_ratings(self, path):
    #     ratings = pd.read_csv(path)
    #     ratings = ratings.rename(columns={'userId':'user', 'movieId':'id'})
    #     ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
    #     return ratings
        
    # def preprocessing_movies(self, path):
    #     items = pd.read_csv(path)
    #     items['id'] = pd.to_numeric(items['id'], errors='coerce')
    #     items = items.dropna(subset=['id'])
    #     items['id'] = items['id'].astype('int64')
    #     items['adult'] = items['adult'].map({'True': True, 'False': False})
    #     items['adult']= items['adult'].astype('bool')
    #     items = items.convert_dtypes()
    #     return items
  
    def similarity_matrix_cosine(self):
        """
        Crea una matriu de similitud entre articles utilitzant la similitud de cosseno.
        """
        ratings_matrix = self.ratings.pivot(index='user', columns='id', values='rating').fillna(0)
        similarity_matrix = cosine_similarity(ratings_matrix.T)  
        self.item_similarity = pd.DataFrame(similarity_matrix, index=ratings_matrix.columns, columns=ratings_matrix.columns)
        
    def similarity_matrix_pearson(self):
        """
        Crea una matriu de similitud entre articles utilitzant la similitud de Pearson.
        """
        ratings_matrix = self.ratings.pivot(index='user', columns='id', values='rating').fillna(0)
        similarity_matrix = ratings_matrix.T.corr(method='pearson')
        self.item_similarity = similarity_matrix
        
    def get_similar_items(self, item_id, similarity,top_n=5):
        """
        Retorna els articles més similars a un article donat.
        :param item_id: ID de la pel·lícula per al qual es volen recomanacions
        :param top_n: Nombre d'articles similars que es volen obtenir
        :return: DataFrame amb els articles més similars
        """
        if self.item_similarity is None:
            if similarity == 'Cosine':
                self.similarity_matrix_cosine()
            elif similarity == 'Pearson':
                self.similarity_matrix_pearson()
            else:
                raise ValueError("Métode desconegut: només 'Cosine' o 'Pearson' són vàlids.")
            
        if item_id not in self.item_similarity.columns:
            print(f"Warning: Article {item_id} no trobat a la matriu de similitud de Pearson.")
            
            return pd.DataFrame()  # Retorna un DataFrame buit si l'article no existeix  
        similarity_scores = self.item_similarity[item_id]
        
        similar_items = similarity_scores.sort_values(ascending=False).head(top_n + 1)  # +1 per incloure's a si mateix
        similar_items = similar_items.drop(item_id) 
        return similar_items
    
    def recommend_for_user(self, user_id, similarity='Cosine',top_n=5):
        """
        Recomana articles a un usuari basat en els articles que ha valorat més altament.
        :param user_id: ID de l'usuari
        :param top_n: Nombre de recomanacions
        :return: DataFrame amb les recomanacions per a l'usuari
        """
        user_ratings = self.ratings[self.ratings['user'] == user_id]
        
        recommendations = {}
        for _, row in user_ratings.iterrows():
            item_id = row['id']
            rating = row['rating']
            
            similar_items = self.get_similar_items(item_id, similarity,top_n=top_n)
            
            for similar_item, score in similar_items.items():
                if similar_item not in recommendations:
                    recommendations[similar_item] = 0
                
                recommendations[similar_item] += score * rating
        
        recommended_items = pd.Series(recommendations).sort_values(ascending=False).head(top_n)
        
        recommended_items_info = self.items.loc[recommended_items.index]
        recommended_items_info['score'] = recommended_items.values
        
        return recommended_items_info

if __name__ == '__main__':
    user_id = 1
    # recommender = ItemItemRecommender()
    # recs_cos = recommender.recommend_for_user(user_id, similarity='Cosine',top_n=3)
    # print('Recomanacions cosinus:\n',recs_cos)
    
    recommender = ItemItemRecommender()
    recs_pear = recommender.recommend_for_user(user_id, similarity='Pearson',top_n=3)
    print('\nRecomanacions pearson:\n',recs_pear)
    