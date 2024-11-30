import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserUserRecommender:
    def __init__(self):
        """
        Inicialitza el sistema de recomanació amb les dades de valoracions i usuaris.
        """
        self.ratings = pd.read_csv('./datasets/ratings_small.csv')
        self.ratings_matrix = None
        self.user_similarity = None
    
        # # Creem un conjunt de dades d'exemple
        # self.ragins = {
        #     'userId': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
        #     'movieId': [110, 147, 858, 1221, 1246, 1968, 2762, 2918, 110, 147, 858, 1221, 1221, 2762],
        #     'rating': [1.0, 4.5, 5.0, 5.0, 5.0, 4.0, 4.5, 5.0, 5.0, 3.0, 4.0, 5.0, 3.5, 4.0]
        # }

    def similarity_matrix(self, method):
        """
        Crea una matriu de similitud entre usuaris utilitzant la similitud de cosseno.
        """
        # Creem la matriu de valoracions per usuaris i pel·lícules
        ratings_matrix = self.ratings.pivot(index='userId', columns='movieId', values='rating')

        # Calculem la similitud entre usuaris mitjançant la similitud de cosinus
        self.ratings_matrix = ratings_matrix.fillna(0)
        
        if method == 'cosine':
            similarity_matrix = cosine_similarity(self.ratings_matrix)
        
        # Calculem la similitud entre usuaris mitjançant Pearson (correlació)
        elif method == 'pearson':
            similarity_matrix = self.ratings_matrix.corr(method=method, min_periods=1)
        
        # Convertim les matrius de similitud en DataFrames per facilitar la consulta
        self.user_similarity = pd.DataFrame(similarity_matrix, index=self.ratings_matrix.index, columns=self.ratings_matrix.index)
        

    # Funció per prediure la valoració
    def predict_rating(self, user_id, movie_id, topN=3, method='cosine'):
        self.similarity_matrix(method)
        
        # Comprovem si la pel·lícula existeix en la matriu de valoracions
        if movie_id not in self.ratings_matrix.columns:
            return np.nan

        # Comprovem si l'usuari existeix en la matriu de valoracions
        if user_id not in self.ratings_matrix.index:
            return np.nan 

        # Obtenir les valoracions de l'usuari
        user_ratings = self.ratings_matrix.loc[user_id]

        # Calcular la similitud entre l'usuari actual i els altres usuaris
        if method == 'cosine':
            similar_users = self.user_similarity.loc[user_id].dropna().sort_values(ascending=False)
        elif method == 'pearson':
            similar_users = self.user_similarity.loc[user_id].dropna().sort_values(ascending=False)
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson' són vàlids.")

        # Seleccionar els K usuaris més semblants
        most_similar_users = similar_users.head(topN).index

        # Calculant la mitjana ponderada normalitzada
        sum_nom = 0
        sum_dem = 0
        mean_user_rating = user_ratings.mean()

        for similar_user in most_similar_users:
            # Comprovar si l'usuari similar existeix a la matriu de valoracions
            if similar_user not in self.ratings_matrix.index:
                continue  # Si l'usuari no existeix, el passem per alt

            user_ratings_neighbour = self.ratings_matrix.loc[similar_user]
            if pd.isna(user_ratings_neighbour[movie_id]):
                continue
            mean_neighbour = user_ratings_neighbour.mean()

            # Ponderar les valoracions dels usuaris similars
            sum_nom += (user_ratings_neighbour[movie_id] - mean_neighbour) * self.user_similarity.loc[user_id, similar_user]
            sum_dem += abs(self.user_similarity.loc[user_id, similar_user])

        if sum_dem != 0:
            pred_rating = mean_user_rating + sum_nom / sum_dem
            return pred_rating
        else:
            return mean_user_rating 


if __name__=="__main__":
    recomender = UserUserRecommender()
    user_id = 1 #Usuari número 1
    movie_id = 1221  # Pel·lícula amb id 1221
    
    # Exemple per a la similitud de cosinus
    predicted_rating_cosine = recomender.predict_rating(user_id, movie_id, topN=3, method='cosine')
    print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")

    # Exemple per a la correlació de Pearson
    predicted_rating_pearson = recomender.predict_rating(user_id, movie_id, topN=3, method='pearson')
    print(f"Predicció de valoració (Pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")
