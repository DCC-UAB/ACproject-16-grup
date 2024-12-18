import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import json


PATH_MOVIES = "./datasets/movies_metadata.csv"
PATH_RATINGS = "./datasets/ratings.csv"
PATH_RATINGS_SMALL = "./datasets/ratings_small.csv"
PATH_CREDITS = "./datasets/credits.csv"
PATH_KEYWORDS = "./datasets/keywords.csv"
PATH_LINKS = "./datasets/links.csv"
PATH_LINKS_SMALL = "./datasets/links_small.csv"

ID = "id"


def movies_metadata(path=PATH_MOVIES):
    items = pd.read_csv(path, low_memory=False)
    items[ID] = pd.to_numeric(items[ID], errors="coerce")
    items = items.dropna(subset=[ID])
    items[ID] = items[ID].astype("int64")

    items["adult"] = items["adult"].map({"True": True, "False": False})
    items["adult"] = items["adult"].astype("bool")

    items["belongs_to_collection"] = items["belongs_to_collection"].fillna("{}")
    items["belongs_to_collection"] = items["belongs_to_collection"].apply(
        ast.literal_eval
    )
    items["belongs_to_collection"] = items["belongs_to_collection"].apply(
        lambda x: None if x == {} else x
    )

    # TODO: genres, production_companies, production_countries, spoken_languages
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


def data_ratings(path=PATH_RATINGS):
    ratings = pd.read_csv(path)
    ratings = ratings.rename(columns={"userId": "user", "movieId": ID})
    ratings.timestamp = pd.to_datetime(ratings.timestamp, unit="s")
    return ratings


def convert_to_dict_list(string):
    return ast.literal_eval(string)



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


def keywords(movies, path=PATH_KEYWORDS):
    keywords = pd.read_csv(path)
    keywords["keywords"] = keywords["keywords"].apply(convert_to_dict_list)
    keywords["keywords_id"] = keywords["keywords"].apply(lambda x: [d["id"] for d in x if isinstance(d, dict) and "id" in d])
    keywords["keywords"] = keywords["keywords"].apply(lambda x: [d["name"] for d in x if isinstance(d, dict) and "name" in d])

    movies_metadata = movies[["id", "overview"]].dropna(subset=["overview"])
    keywords = keywords[keywords["id"].isin(movies_metadata["id"])]
    df = pd.merge(keywords, movies_metadata, on="id", how="left")
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
            else:
                keywords_for_movie = []
            keywords.at[keywords[keywords["id"] == row['id']].index[0], "keywords"] = keywords_for_movie
    return keywords


def links(path=PATH_LINKS):
    return pd.read_csv(path)


def small_ratings():
    ratings = data_ratings(PATH_RATINGS_SMALL)
    movies = movies_metadata(PATH_MOVIES)
    movies = movies[movies[ID].isin(ratings[ID])]
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
    key = keywords(movies, PATH_KEYWORDS)
    #ground, ratingst = ground_truth(rates)
