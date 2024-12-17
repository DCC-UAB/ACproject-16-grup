import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


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


def credits(path, path_movies=PATH_MOVIES):
    movies = movies_metadata(path_movies)
    credits = pd.read_csv(path)

    id_movies = set(movies[ID])

    credits = credits[credits[ID].isin(id_movies)]

    credits["cast"] = credits["cast"].apply(convert_to_dict_list)
    credits["crew"] = credits["crew"].apply(convert_to_dict_list)

    cast_rows = []
    crew_rows = []

    for _, row in credits.iterrows():
        for cast_member in row.cast:
            cast_member["movie_id"] = row[ID]
            cast_rows.append(cast_member)

        for crew_member in row.crew:
            crew_member["movie_id"] = row[ID]
            crew_rows.append(crew_member)

    cast = pd.DataFrame(cast_rows)
    crew = pd.DataFrame(crew_rows)

    return cast.convert_dtypes(), crew.convert_dtypes()


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
    return ratings, movies, key


if __name__ == "__main__":
    # cast, crew = credits(PATH_CREDITS, PATH_MOVIES)
    rates, movies, key= small_ratings()
    key = keywords(movies, PATH_KEYWORDS)
    print(key["keywords"].tail(10))
    # links = links(PATH_LINKS)
    # links_small = links(PATH_LINKS_SMALL)
    # movies = movies_metadata(PATH_MOVIES)
    # ratings = data_ratings(PATH_RATINGS)
    

    # print('CREDITS\n')
    # print(cast.info())
    # time.sleep(5)
    # print(crew.info())
    # time.sleep(5)

    # print('\n\nKEYWORDS\n',)
    # print(keywords.info())
    # time.sleep(5)

    # print('\n\nLINKS\n')
    # print(links.info())
    # time.sleep(5)

    # print('\n\nLINKS_SMALL\n',)
    # print(links_small.info())
    # time.sleep(5)

    # print('\n\nMOVIES\n')
    # print(movies.info())
    # time.sleep(5)

    # print('\n\nRATINGS\n')
    # print(ratings.info())
    # time.sleep(5)

    # print('\n\nRATINGS_SMALL\n')
    # print(ratings_small.info())
    # time.sleep(5)
