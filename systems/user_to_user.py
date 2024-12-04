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
            
            # Creem la matriu de ratings (usuaris x pel·lícules)
            self.__ratings_matrix = self.__ratings.pivot_table(index='userId', columns='movieId', values='rating')
            
            # Reemplaçar els valors NaN amb 0 per la similitud de cosinus
            self.__ratings_matrix = self.__ratings_matrix.fillna(0)
            
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

    def similarity_matrix(self, method='cosine'):
        """
        Crea una matriu de similitud entre usuaris utilitzant la similitud de cosseno.
        """
        # Creem la matriu de valoracions per usuaris i pel·lícules
        ratings_matrix = self.__ratings.pivot(index='userId', columns='movieId', values='rating')

        # Calculem la similitud entre usuaris mitjançant la similitud de cosinus
        self.__ratings_matrix = ratings_matrix.fillna(0)
        
        if method == 'cosine':
            similarity_matrix = cosine_similarity(self.__ratings_matrix)
        
        # Calculem la similitud entre usuaris mitjançant Pearson (correlació)
        elif method == 'pearson':
            similarity_matrix = self.__ratings_matrix.T.corr(method='pearson')
        
        # Convertim les matrius de similitud en DataFrames per facilitar la consulta
        self.__user_similarity = pd.DataFrame(similarity_matrix, index=self.__ratings_matrix.index, columns=self.__ratings_matrix.index)
        

    # Funció per prediure la valoració
    def predict_rating(self, user_id, movie_id, topN=20, method='cosine'):
        self.similarity_matrix(method)
        
        # Comprovem si la pel·lícula existeix en la matriu de valoracions
        if movie_id not in self.__ratings_matrix.columns:
            return np.nan

        # Comprovem si l'usuari existeix en la matriu de valoracions
        if user_id not in self.__ratings_matrix.index:
            return np.nan 

        # Obtenir les valoracions de l'usuari
        user_ratings = self.__ratings_matrix.loc[user_id]

        # Calcular la similitud entre l'usuari actual i els altres usuaris
        if method == 'cosine':
            similar_users = self.__user_similarity.loc[user_id].dropna().sort_values(ascending=False)
        elif method == 'pearson':
            similar_users = self.__user_similarity.loc[user_id].dropna().sort_values(ascending=False)
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson' són vàlids.")

        # Seleccionar els K usuaris més semblants
        most_similar_users = similar_users.head(topN).index
        print(f"Most similar users using {method}:",most_similar_users)

        # Calculant la mitjana ponderada normalitzada
        sum_nom = 0
        sum_dem = 0
        mean_user_rating = user_ratings.mean()

        for similar_user in most_similar_users:
            # Comprovar si l'usuari similar existeix a la matriu de valoracions
            if similar_user not in self.__ratings_matrix.index:
                continue  # Si l'usuari no existeix, el passem per alt

            user_ratings_neighbour = self.__ratings_matrix.loc[similar_user]
            if pd.isna(user_ratings_neighbour[movie_id]):
                continue
            mean_neighbour = user_ratings_neighbour.mean()

            # Ponderar les valoracions dels usuaris similars
            sum_nom += (user_ratings_neighbour[movie_id] - mean_neighbour) * self.__user_similarity.loc[user_id, similar_user]
            sum_dem += abs(self.__user_similarity.loc[user_id, similar_user])

        if sum_dem != 0:
            pred_rating = mean_user_rating + sum_nom / sum_dem
            return pred_rating
        else:
            return mean_user_rating 

    def generate_recommendation_matrix(self, topN=10, method='cosine'):
        """
        Genera una matriu de recomanacions per a tots els usuaris i pel·lícules.
        Per defecte, els 20 més alts.
        """
        self.similarity_matrix(method)
        recommendation_matrix = {}

        for user_id in self.__ratings_matrix.index:
            recommendation_matrix[user_id] = {}
            for movie_id in self.__ratings_matrix.columns:
                # Si l'usuari ja ha valorat la pel·lícula, no la recomanem
                if pd.notna(self.__ratings_matrix.loc[user_id, movie_id]):
                    continue

                # Predim la valoració per a aquesta pel·lícula
                pred_rating = self.predict_rating(user_id, movie_id, topN=topN, method=method)
                recommendation_matrix[user_id][movie_id] = pred_rating

                # Depuració per veure el que estem predint
                print(f"Usuari {user_id} - Pel·lícula {movie_id} -> Predicció: {pred_rating}")

        print("Matriu de recomanacions:")
        for user, recommendations in recommendation_matrix.items():
            print(f"Usuari {user}: {recommendations}")

        return recommendation_matrix
    


if __name__ == "__main__":
    recommender = UserUserRecommender()
    recommender.load_data('./datasets/ratings_small.csv')

    # Exemple per a la predicció de valoració utilitzant la similitud de cosinus
    user_id = 1  # Usuari número 1
    movie_id = 31  # Pel·lícula amb id 1221
    
    predicted_rating_cosine = recommender.predict_rating(user_id, movie_id, topN=20, method='cosine')
    print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")

    # Exemple per a la predicció de valoració utilitzant la correlació de Pearson
    predicted_rating_pearson = recommender.predict_rating(user_id, movie_id, topN=20, method='pearson')
    print(f"Predicció de valoració (Pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")

    # Generar la matriu de recomanacions per a tots els usuaris
    recommendation_matrix = recommender.generate_recommendation_matrix(topN=3, method='cosine')

    # Mostrar els resultats de la matriu de recomanacions
    print("\nMatriu de recomanacions:")
    for user_id, recommendations in recommendation_matrix.items():
        print(f"Usuari {user_id}:")
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        for movie_id, predicted_rating in sorted_recommendations:
            print(f"  Pel·lícula {movie_id}: {predicted_rating:.2f}")




    '''
    # Prova de la matriu de recomanacions per a tots els usuaris
    recommendation_matrix = recomender.generate_recommendation_matrix(topN=5, method='cosine')
    
    # Mostrem la matriu de recomanacions per a la depuració
    #print("\nMatriu de recomanacions:")
    #for user_id, recommendations in recommendation_matrix.items():
    #    print(f"\nUsuari {user_id}:")
    #    for movie_id, rating in recommendations.items():
    #        print(f"Pel·lícula ID: {movie_id}, Predicció de valoració: {rating}")

    # Mostrem les recomanacions per alguns usuaris (per exemple, usuaris 1 i 2)
    print("\nRecomanacions per a l'usuari 1:")
    for movie_id, rating in recommendation_matrix[1].items():
        print(f"Pel·lícula ID: {movie_id}, Predicció de valoració: {rating}")

    print("\nRecomanacions per a l'usuari 2:")
    for movie_id, rating in recommendation_matrix[2].items():
        print(f"Pel·lícula ID: {movie_id}, Predicció de valoració: {rating}")
    '''