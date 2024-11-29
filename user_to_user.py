import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Creem un conjunt de dades d'exemple
data = {
    'userId': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
    'movieId': [110, 147, 858, 1221, 1246, 1968, 2762, 2918, 110, 147, 858, 1221, 1221, 2762],
    'rating': [1.0, 4.5, 5.0, 5.0, 5.0, 4.0, 4.5, 5.0, 5.0, 3.0, 4.0, 5.0, 3.5, 4.0]
}

df = pd.read_csv('./datasets/ratings_small.csv')

#df = pd.DataFrame(data)

# Creem la matriu de valoracions per usuaris i pel·lícules
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Calculem la similitud entre usuaris mitjançant la similitud de cosinus
ratings_matrix_filled = ratings_matrix.fillna(0)
similarity_matrix_cosine = cosine_similarity(ratings_matrix_filled)

# Calculem la similitud entre usuaris mitjançant Pearson (correlació)
similarity_matrix_pearson = ratings_matrix.corr(method='pearson', min_periods=1)

# Convertim les matrius de similitud en DataFrames per facilitar la consulta
similarity_matrix_cosine = pd.DataFrame(similarity_matrix_cosine, index=ratings_matrix.index, columns=ratings_matrix.index)
similarity_matrix_pearson = pd.DataFrame(similarity_matrix_pearson, index=ratings_matrix.index, columns=ratings_matrix.index)

# Funció per prediure la valoració
def predict_rating(user_id, movie_id, similarity_matrix, ratings_matrix, topN=3, method='cosine'):
    # Comprovem si la pel·lícula existeix en la matriu de valoracions
    if movie_id not in ratings_matrix.columns:
        return np.nan

    # Comprovem si l'usuari existeix en la matriu de valoracions
    if user_id not in ratings_matrix.index:
        return np.nan 

    # Obtenir les valoracions de l'usuari
    user_ratings = ratings_matrix.loc[user_id]

    # Calcular la similitud entre l'usuari actual i els altres usuaris
    if method == 'cosine':
        similar_users = similarity_matrix.loc[user_id].dropna().sort_values(ascending=False)
    elif method == 'pearson':
        similar_users = similarity_matrix.loc[user_id].dropna().sort_values(ascending=False)
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

# Exemple d'ús per a la similitud de cosinus
user_id = 1
movie_id = 1221  # Pel·lícula amb id 1221
predicted_rating_cosine = predict_rating(user_id, movie_id, similarity_matrix_cosine, ratings_matrix, topN=3, method='cosine')
print(f"Predicció de valoració (cosinus) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_cosine:.2f}")

# Exemple d'ús per a la similitud de Pearson
predicted_rating_pearson = predict_rating(user_id, movie_id, similarity_matrix_pearson, ratings_matrix, topN=3, method='pearson')
print(f"Predicció de valoració (Pearson) per l'usuari {user_id} per la pel·lícula {movie_id}: {predicted_rating_pearson:.2f}")
