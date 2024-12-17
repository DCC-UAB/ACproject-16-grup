import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time

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


    def predict_rating(self, user_id, movie_id, topN=3, method='cosine'):
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

    def split_data(self, min_user_ratings=5, min_movie_ratings=5, train_size=0.7, validation_size=0.15, random_state=42):
        """
        Divideix el dataset en tres subconjunts: entrenament, validació i test.
        """
        # Assegura't que 'userId' i 'movieId' són columnes normals
        self.__ratings = self.__ratings.reset_index(drop=True)

        # Filtra usuaris i pel·lícules amb suficients valoracions
        user_counts = self.__ratings.groupby('userId').size()
        valid_users = user_counts[user_counts >= min_user_ratings].index

        movie_counts = self.__ratings.groupby('movieId').size()
        valid_movies = movie_counts[movie_counts >= min_movie_ratings].index

        filtered_ratings = self.__ratings[
            (self.__ratings['userId'].isin(valid_users)) &
            (self.__ratings['movieId'].isin(valid_movies))
        ]

        # Barreja les dades i divideix-les
        shuffled_ratings = filtered_ratings.sample(frac=1, random_state=random_state)

        train_end = int(train_size * len(shuffled_ratings))
        validation_end = train_end + int(validation_size * len(shuffled_ratings))

        train_data = shuffled_ratings.iloc[:train_end]
        validation_data = shuffled_ratings.iloc[train_end:validation_end]
        test_data = shuffled_ratings.iloc[validation_end:]

        return train_data, validation_data, test_data

    def recommend_movies_for_user(self, user_id, topN=10, method='cosine'):
        """
        Recomana les millors pel·lícules per a un usuari que encara no hagi vist.
        """
        self.similarity_matrix(method)
        
        if user_id not in self.__ratings_matrix.index:
            print(f"L'usuari {user_id} no té dades al sistema.")
            return []
        
        user_ratings = self.__ratings_matrix.loc[user_id]
        unseen_movies = user_ratings[user_ratings.isna()].index  # Pel·lícules no valorades per l'usuari
        
        # Predir les valoracions per a totes les pel·lícules no vistes
        predictions = {
            movie_id: self.predict_rating(user_id, movie_id, topN=topN, method=method)
            for movie_id in unseen_movies
        }
        
        # Ordenar les pel·lícules per la valoració predita, de més alta a més baixa
        recommended_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:topN]
        return recommended_movies


if __name__ == "__main__":
    start_time = time.time()
    recommender = UserUserRecommender()
    recommender.load_data('./datasets/ratings_small.csv')

     # Divideix les dades en train, validation, i test
    train_data, validation_data, test_data = recommender.split_data(
        min_user_ratings=5, min_movie_ratings=5, train_size=0.7, validation_size=0.15
    )
    
    print(f"Conjunt d'entrenament: {len(train_data)} valoracions")
    print(f"Conjunt de validació: {len(validation_data)} valoracions")
    print(f"Conjunt de test: {len(test_data)} valoracions")

    # Carregar només les dades d'entrenament al model
    recommender._UserUserRecommender__ratings = train_data

    # Predir les valoracions del conjunt de validació
    validation_data['predicted_rating'] = validation_data.apply(
        lambda row: recommender.predict_rating(row['userId'], row['movieId'], topN=20, method='cosine'), axis=1
    )
    
    # Càlcul de l'error en validació
    mae_validation = (validation_data['rating'] - validation_data['predicted_rating']).abs().mean()
    print(f"Error MAE en validació: {mae_validation}")

    rmse_validation = np.sqrt(((validation_data['rating'] - validation_data['predicted_rating']) ** 2).mean())
    print(f"Error RMSE en validació: {rmse_validation}")

    # Predir les valoracions del conjunt de test
    test_data['predicted_rating'] = test_data.apply(
        lambda row: recommender.predict_rating(row['userId'], row['movieId'], topN=20, method='cosine'), axis=1
    )
    
    # Càlcul de l'error en test
    mae_test = (test_data['rating'] - test_data['predicted_rating']).abs().mean()
    print(f"Error MAE en test: {mae_test}")

    rmse_test = np.sqrt(((test_data['rating'] - test_data['predicted_rating']) ** 2).mean())
    print(f"Error RMSE en test: {rmse_test}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Temps total: {elapsed_time:.2f} segons")

    while True:
        try:
            user_id = int(input("Introdueix l'ID de l'usuari per qui vols recomanacions (o -1 per sortir): "))
            if user_id == -1:
                break

            recommended_movies = recommender.recommend_movies_for_user(user_id, topN=10, method='cosine')

            if not recommended_movies:
                print(f"No s'han trobat pel·lícules per recomanar a l'usuari {user_id}.")
            else:
                print(f"Les 10 millors pel·lícules recomanades per l'usuari {user_id} són:")
                for i, (movie_id, predicted_rating) in enumerate(recommended_movies, start=1):
                    print(f"{i}. Pel·lícula ID {movie_id} - Valoració predita: {predicted_rating:.2f}")
        except ValueError:
            print("Si us plau, introdueix un ID vàlid.")    






    '''
    # Exemple per a la predicció
    user_id = 402
    movie_id = 13

    predicted_rating_cosine = recommender.predict_rating(user_id, movie_id, topN=20, method='cosine')
    print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine}")

    predicted_rating_pearson = recommender.predict_rating(user_id, movie_id, topN=20, method='pearson')
    print(f"Predicció de valoració (pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson}")

    # Generar la matriu de recomanacions
    recommendation_matrix = recommender.generate_recommendation_matrix(topN=10, method='cosine')
    print("\nMatriu de recomanacions:\n", recommendation_matrix)
    '''