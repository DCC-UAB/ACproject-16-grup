import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserUserRecommender:
    def __init__(self):
        """
        Inicialitza el sistema de recomanació amb les dades de valoracions i usuaris.
        """
        self.__ratings = None
        self.__ratings_matrix = None
        self.__user_similarity = None
    
    def load_data(self, file_path):
        try:
            self.__ratings = pd.read_csv(file_path)
            if self.__ratings.isnull().values.any():
                print("Atenció: Hi ha valors nuls a les dades carregades.")
            
            # Creem la matriu de ratings (usuaris x pel·lícules) com un DataFrame
            self.__ratings_matrix = self.__ratings.pivot_table(index='userId', columns='movieId', values='rating')

            # Normalitzem les valoracions per usuari abans de la matriu de similitud
            self.__ratings = self.normalize_ratings(self.__ratings)

            # Omplim NaN amb la mitjana de cada usuari
            #self.__ratings_matrix = self.__ratings_matrix.apply(lambda col: col.fillna(col.mean()), axis=0)
            self.__ratings_matrix = self.__ratings_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)
            #self.__ratings_matrix = self.__ratings_matrix.apply(lambda row: row.fillna(0), axis=1)

            
            print(self.__ratings.head(10))  # Mostrem les primeres 10 files per depurar
            self.similarity_matrix()  # Cridem la funció per generar la matriu de similitud
            
        except Exception as e:
            print(f"Error carregant el fitxer: {e}")

    @property
    def ratings_matrix(self):
        return self.__ratings_matrix
    
    @property
    def user_similarity(self):
        return self.__user_similarity

    def normalize_ratings(self, ratings):
        """
        Normalitza les valoracions per usuari (centrant-les a zero i escalant-les per la desviació estàndard).
        """
        print("Normalitzant les valoracions per usuari...")

        def normalize_user(group):
            mean = group['rating'].mean()
            std = group['rating'].std() if group['rating'].std() != 0 else 1
            group['rating_normalized'] = (group['rating'] - mean) / std
            return group

        ratings = ratings.groupby('userId').apply(normalize_user)
        return ratings

    def similarity_matrix(self, method='cosine'):
        """
        Crea una matriu de similitud entre usuaris utilitzant pandas i similitud de cosinus.
        """
        if method == 'cosine':
            # Similitud de cosinus
            similarity_matrix = cosine_similarity(self.__ratings_matrix)
            self.__user_similarity = pd.DataFrame(
                similarity_matrix, 
                index=self.__ratings_matrix.index, 
                columns=self.__ratings_matrix.index
            )
        elif method == 'pearson':
            # Similitud de Pearson com una correlació
            self.__user_similarity = self.__ratings_matrix.T.corr(method='pearson')
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson' són vàlids.")
        
        print("Matriu de similitud entre usuaris:")
        print(self.__user_similarity)

    def predict_rating(self, user_id, movie_id, topN=20, method='cosine'):
        self.similarity_matrix(method)
        
        if movie_id not in self.__ratings_matrix.columns or user_id not in self.__ratings_matrix.index:
            return np.nan

        similar_users = self.__user_similarity.loc[user_id].drop(user_id).sort_values(ascending=False).head(topN)
        user_ratings = self.__ratings_matrix.loc[user_id]

        # SOLUCIÓ 3: Ajustem el càlcul de prediccions
        numerator = sum(
            self.__ratings_matrix.loc[sim_user, movie_id] * similarity
            for sim_user, similarity in similar_users.items()
            if pd.notna(self.__ratings_matrix.loc[sim_user, movie_id])
        )
        denominator = similar_users.abs().sum()

        # SOLUCIÓ 2: Evitem divisió per zero
        if denominator == 0:
            return user_ratings.mean()
        
        return numerator / denominator

    def generate_recommendation_matrix(self, topN=10, method='cosine'):
        """
        Genera una matriu de recomanacions en format DataFrame.
        """
        self.similarity_matrix(method)
        
        # Creem una còpia de la matriu per no modificar l'original
        recommendations = self.__ratings_matrix.copy()
        
        # Iterem pels usuaris i les pel·lícules per generar prediccions
        for user_id in self.__ratings_matrix.index:
            for movie_id in self.__ratings_matrix.columns:
                if pd.isna(self.__ratings_matrix.loc[user_id, movie_id]):  # Si no hi ha valoració
                    pred_rating = self.predict_rating(user_id, movie_id, topN=topN, method=method)
                    recommendations.loc[user_id, movie_id] = pred_rating  # Omplim amb la predicció
        
        return recommendations



if __name__ == "__main__":
    recommender = UserUserRecommender()
    recommender.load_data('./datasets/ratings_small.csv')

    # Exemple per a la predicció
    user_id = 1
    movie_id = 30

    predicted_rating_cosine = recommender.predict_rating(user_id, movie_id, topN=20, method='cosine')
    print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")

    predicted_rating_pearson = recommender.predict_rating(user_id, movie_id, topN=20, method='pearson')
    print(f"Predicció de valoració (pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")

    # Generar la matriu de recomanacions
    recommendation_matrix = recommender.generate_recommendation_matrix(topN=10, method='cosine')
    print("\nMatriu de recomanacions:\n", recommendation_matrix)
