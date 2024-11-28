import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

#La base de dades és massa gran, creo un exemple per veure funcionalitat
data = {
    'userId': [1, 1, 1, 1, 1, 1, 1, 1],
    'movieId': [110, 147, 858, 1221, 1246, 1968, 2762, 2918],
    'rating': [1.0, 4.5, 5.0, 5.0, 5.0, 4.0, 4.5, 5.0]
}

df = pd.DataFrame(data)


#ratings = pd.read_csv("datasets/ratings.csv")


# Funció que et pivota les dades per tenir les pel·lícules com a columnes i els usuaris com a files
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Utilitzant Pearson # Mínim 10 valoracions comunes, sinó la matriu serà enorme. Augmentar en cas necessari
similarity_matrix_pearson = ratings_matrix.corr(method='pearson', min_periods=10)  

# Calcular la similitud entre usuaris utilitzant la similitud de cosinus. No podem tenir NaN en el cosinus
ratings_matrix_filled = ratings_matrix.fillna(0) 

similarity_matrix_cosine = cosine_similarity(ratings_matrix_filled)

# Convertim la similitud de cosinus a un DataFrame amb els mateixos índexs i columnes per facilitar la consulta
similarity_matrix_cosine = pd.DataFrame(similarity_matrix_cosine, index=ratings_matrix.index, columns=ratings_matrix.index)


# Predicció valoració d'una pel·lícula per un usuari - Fet a classe
def predict_rating(user_id, movie_id, similarity_matrix, ratings_matrix, topN=3, method='pearson'):
    if movie_id not in ratings_matrix.columns:
        return np.nan  # Si la pel·lícula no està en el conjunt de dades, no podem fer la predicció
    
    # Comprovar si l'usuari existeix en la matriu de valoracions
    if user_id not in ratings_matrix.index:
        return np.nan  # Si l'usuari no està en la matriu, retornem NaN

    # Obtenir les valoracions de l'usuari i les dels seus veïns més semblants
    user_ratings = ratings_matrix.loc[user_id]
    
    if method == 'pearson':
        similar_users = similarity_matrix[movie_id].dropna().sort_values(ascending=False)
    elif method == 'cosine':
        similar_users = similarity_matrix[movie_id].dropna().sort_values(ascending=False)
    else:
        raise ValueError("Métode desconegut: només 'pearson' o 'cosine' són vàlids.")
    
    # Seleccionar els K usuaris més semblants
    most_similar_users = similar_users.head(topN).index
    
    # Calculant la mitjana ponderada normalitzada
    sum_nom = 0
    sum_dem = 0
    mean_user_rating = user_ratings.mean()

    for similar_user in most_similar_users:
        # Comprovar si l'usuari similar existeix a la matriu de valoracions
        if similar_user not in ratings_matrix.index:
            continue  # Si l'usuari no existeix, el passem per alt

        user_ratings_neighbour = ratings_matrix.loc[similar_user]
        if pd.isna(user_ratings_neighbour[movie_id]):
            continue
        mean_neighbour = user_ratings_neighbour.mean()
        
        # Ponderar les valoracions dels usuaris similars
        sum_nom += (user_ratings_neighbour[movie_id] - mean_neighbour) * similarity_matrix.loc[user_id, similar_user]
        sum_dem += abs(similarity_matrix.loc[user_id, similar_user])
    
    if sum_dem != 0:
        pred_rating = mean_user_rating + sum_nom / sum_dem
        return pred_rating
    else:
        return mean_user_rating  # Si no es pot calcular, retornem la mitjana de l'usuari


# Exemple d'ús
userId = 1
movieId = 147

predicted_rating = predict_rating(userId, movieId, similarity_matrix_cosine, ratings_matrix, topN=3)
print(f"La valoració predicha per l'usuari {userId} per la pel·lícula {movieId} és: {predicted_rating}")


