import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json
import nltk
import csv

PATH_MOVIES = "./datasets/movies_metadata.csv"
PATH_RATINGS = "./datasets/ratings.csv"
PATH_RATINGS_SMALL = "./datasets/ratings_small.csv"
PATH_CREDITS = "./datasets/credits.csv"
PATH_KEYWORDS = "./datasets/keywords.csv"
PATH_LINKS = "./datasets/links.csv"
PATH_LINKS_SMALL = "./datasets/links_small.csv"

def read_binary_csv(path):
    with open(path, 'rb') as fitxer_binari:
        binary_data = fitxer_binari.read()
    descodificacio = binary_data.decode('utf-8')  
    reader = csv.reader(descodificacio.splitlines(), delimiter=',')
    dades = [fila for fila in reader]

    header = dades[0]  
    dades_sense_capcalera = dades[1:]
    df = pd.DataFrame(dades_sense_capcalera, columns=header)
    df["userId"] = df["userId"].astype(int)
    df["movieId"] = df["movieId"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["timestamp"] = df["timestamp"].astype(int)
    return df

def movies_metadata(path=PATH_MOVIES):
    items = pd.read_csv(path, low_memory=False)

    # Processament de la columna 'id'
    items["id"] = pd.to_numeric(items["id"], errors="coerce")
    items = items.dropna(subset=["id"])
    items["id"] = items["id"].astype("int64")

    # Processament de la columna 'title'
    items = items.dropna(subset=["title"])
    items = items.drop_duplicates(subset=["title"])

    # Processament de la columna 'adult'
    items["adult"] = items["adult"].map({"True": True, "False": False})
    items["adult"] = items["adult"].astype("bool")

    # Processament de la columna 'belongs_to_collection'
    items["belongs_to_collection"] = items["belongs_to_collection"].fillna("{}")
    items["belongs_to_collection"] = items["belongs_to_collection"].apply(
        ast.literal_eval
    )
    items["belongs_to_collection"] = items["belongs_to_collection"].apply(
        lambda x: None if x == {} else x
    )
    
    # Processament columnes: genres, production_companies, production_countries, spoken_languages
    items["genres"] = (
        items["genres"]
        .fillna("[]")
        .apply(ast.literal_eval)
        .apply(lambda x: None if x == {} else x)
    )
    items["production_companies"] = (
        items["production_companies"].fillna("[]").apply(ast.literal_eval)
    )
    items["production_countries"] = (
        items["production_countries"].fillna("[]").apply(ast.literal_eval)
    )
    items["spoken_languages"] = (
        items["spoken_languages"].fillna("[]").apply(ast.literal_eval)
    )

    items = items.convert_dtypes()
    return items


def data_ratings(movies, path=PATH_RATINGS):
    #ratings = pd.read_csv(path)
    ratings = read_binary_csv(path)
    ratings = ratings.rename(columns={"userId": "user", "movieId": "id"}) # Renombrar columnes
    ratings = ratings[ratings["id"].isin(movies["id"])] # Filtrar pel·lícules vàlides
    ratings.timestamp = pd.to_datetime(ratings.timestamp, unit="s") # Convertir timestamp a datetime
    return ratings


def credits(movies, path=PATH_CREDITS):
    credits = pd.read_csv(path)
    id_movies = set(movies['id']) 
    credits = credits[credits['id'].isin(id_movies)]

    def extract_actors(cast_column):
        try:
            cast_list = json.loads(cast_column.replace("'", "\""))
            return ', '.join([actor['name'] for actor in cast_list])
        except (ValueError, TypeError):
            return ''
    
    def extract_director(crew_column):
        try:
            crew_list = json.loads(crew_column.replace("'", "\""))
            for member in crew_list:
                if member['job'] == 'Director':
                    return member['name']
            return ''
        except (ValueError, TypeError):
            return ''

    credits['actors'] = credits['cast'].apply(lambda x: extract_actors(x))
    credits['director'] = credits['crew'].apply(lambda x: extract_director(x))
    credits_cleaned = credits[['id', 'actors', 'director']]
    credits_cleaned = credits_cleaned[(credits_cleaned['actors'] != '') | (credits_cleaned['director'] != '')]
    return credits_cleaned


def convert_to_dict_list(keywords_string):
    try:
        return eval(keywords_string) if isinstance(keywords_string, str) else []
    except:
        return []

def keywords(movies, path):
    PS = nltk.stem.PorterStemmer()
    keywords_df = pd.read_csv(path)

    # Processament de la columna 'keywords'
    keywords_df["keywords"] = keywords_df["keywords"].apply(convert_to_dict_list)
    keywords_df["keywords_id"] = keywords_df["keywords"].apply(lambda x: [d["id"] for d in x if isinstance(d, dict) and "id" in d])
    keywords_df["keywords"] = keywords_df["keywords"].apply(lambda x: [d["name"] for d in x if isinstance(d, dict) and "name" in d])

    # Filtratge de pel·lícules vàlides
    movies_metadata = movies[["id", "overview"]].dropna(subset=["overview"])
    keywords_df = keywords_df[keywords_df["id"].isin(movies_metadata["id"])]
    
    df = pd.merge(keywords_df, movies_metadata, on="id", how="left")
    keywords_roots = dict()
    keywords_select = dict()
    category_keys = []

    # Aplicació de stemming i normalització a les paraules clau
    for idx, row in df.iterrows():
        current_keywords = row['keywords']
        for keyword in current_keywords:
            keyword = keyword.lower()
            root = PS.stem(keyword)
            if root in keywords_roots:
                keywords_roots[root].add(keyword)
            else:
                keywords_roots[root] = {keyword}

    # Normalització de les paraules clau
    for root, words in keywords_roots.items():
        if len(words) > 1:
            selected_keyword = min(words, key=len)  # Triar la paraula més curta (arrel)
        else:
            selected_keyword = list(words)[0]
        category_keys.append(selected_keyword)
        keywords_select[root] = selected_keyword

    # Afegir keywords a les pel·lícules sense 
    no_key = df[df['keywords'].apply(len) == 0].copy()
    if not no_key.empty:
        for idx, row in no_key.iterrows():
            overview = row['overview'] if pd.notnull(row['overview']) and row['overview'].strip() else ''
            if overview:
                tfidf = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
                tfidf_matrix = tfidf.fit_transform([overview])
                tfidf_feature_names = np.array(tfidf.get_feature_names_out())
                tfidf_scores = tfidf_matrix.toarray().flatten()
                top_indices = tfidf_scores.argsort()[-5:][::-1]
                keywords_for_movie = tfidf_feature_names[top_indices].tolist()

                # Normalitzar les noves keywords
                normalized_keywords = []
                for keyword in keywords_for_movie:
                    keyword = keyword.lower()
                    root = PS.stem(keyword)
                    if root in keywords_select:
                        normalized_keywords.append(keywords_select[root])
                    else:
                        normalized_keywords.append(keyword)
                        keywords_select[root] = keyword

                df.at[idx, "keywords"] = normalized_keywords

    # Normalitzar les keywords de les pel·lícules
    df["keywords"] = df["keywords"].apply(lambda kw_list: [keywords_select[PS.stem(kw.lower())] for kw in kw_list])
    return df


def links(path=PATH_LINKS):
    return pd.read_csv(path)


def small_ratings(): 
    movies = movies_metadata(PATH_MOVIES)
    ratings = data_ratings(movies, PATH_RATINGS)
    movies = movies[movies["id"].isin(ratings["id"])]
    key = keywords(movies, PATH_KEYWORDS)
    castcrew = credits(movies, PATH_CREDITS)
    return ratings, movies, key, castcrew


def ground_truth(ratings):
    user_counts = ratings['user'].value_counts()
    valid_users = user_counts[user_counts > 3].index

    filtered_ratings = ratings[ratings['user'].isin(valid_users)]

    sample_size = int(len(ratings) * 0.15)
    ground_truth_sample = filtered_ratings.sample(n=sample_size, random_state=42)

    ratings = ratings.drop(ground_truth_sample.index)

    ground_truth_df = ground_truth_sample[['user', 'id', 'rating']]
    return ground_truth_df, ratings


if __name__ == "__main__":
    rates, movies, key, credit= small_ratings()
    # ground, ratings = ground_truth(rates)
    #ratings = read_binary_csv('./datasets/ratings.csv')
    #ratings = data_ratings(movies_metadata(PATH_MOVIES), './datasets/ratings.csv')
    print(rates.head())
    print(rates.shape)
