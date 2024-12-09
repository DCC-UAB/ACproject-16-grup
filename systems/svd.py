from surprise import SVD, Dataset, Reader, train_test_split

# Carrega les dades
ratings = pd.read_csv("../datasets/ratings_small.csv")

# Definim el format de les dades
reader = Reader(rating_scale=(0.5, 5))  # Escala de valoracions
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# Dividim les dades en conjunt d'entrenament i test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# Inicialitzem el model SVD
model = SVD(n_factors=50, random_state=42)  # `n_factors` defineix el nombre de factors latents

# Entrenem el model
model.fit(trainset)

# Fem prediccions per al conjunt de test
predictions = model.test(testset)

# Exemples de recomanacions per a un usuari concret
user_id = 10
user_ratings = ratings[ratings['userId'] == user_id]['movieId'].values
all_movie_ids = ratings['movieId'].unique()

# Excloem pel·lícules ja valorades
unrated_movies = [movie_id for movie_id in all_movie_ids if movie_id not in user_ratings]

# Prediccions per a pel·lícules no valorades
predicted_scores = [(movie_id, model.predict(user_id, movie_id).est) for movie_id in unrated_movies]
predicted_scores.sort(key=lambda x: x[1], reverse=True)

# Mostrem les millors recomanacions
top_recommendations = predicted_scores[:10]
print("Top recomanacions:")
for movie_id, score in top_recommendations:
    print(f"Pel·lícula ID: {movie_id}, Predicció: {score:.2f}")

