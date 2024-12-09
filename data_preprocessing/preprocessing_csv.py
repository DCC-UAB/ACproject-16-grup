import pandas as pd
import ast
import time

PATH_MOVIES = './datasets/movies_metadata.csv'
PATH_RATINGS = './datasets/ratings.csv'
PATH_RATINGS_SMALL = './datasets/ratings_small.csv'
PATH_CREDITS = './datasets/credits.csv'
PATH_KEYWORDS = './datasets/keywords.csv'
PATH_LINKS = './datasets/links.csv'
PATH_LINKS_SMALL = './datasets/links_small.csv'

ID = 'id'

def movies_metadata(path=PATH_MOVIES):
        items = pd.read_csv(path, low_memory=False)
        items[ID] = pd.to_numeric(items[ID], errors='coerce')
        items = items.dropna(subset=[ID])
        items[ID] = items[ID].astype('int64')
        
        items['adult'] = items['adult'].map({'True': True, 'False': False})
        items['adult']= items['adult'].astype('bool')
        
        items['belongs_to_collection'] = items['belongs_to_collection'].fillna('{}')
        items['belongs_to_collection'] = items['belongs_to_collection'].apply(ast.literal_eval)
        items['belongs_to_collection'] = items['belongs_to_collection'].apply(lambda x: None if x == {} else x)
        
        # TODO: genres, production_companies, production_countries, spoken_languages
        items['genres'] = items['genres'].fillna('[]').apply(ast.literal_eval)
        items['production_companies'] = items['production_companies'].fillna('[]').apply(ast.literal_eval)
        items['production_countries'] = items['production_countries'].fillna('[]').apply(ast.literal_eval)
        items['spoken_languages'] = items['spoken_languages'].fillna('[]').apply(ast.literal_eval)
        
        items = items.convert_dtypes()
        return items
    
def data_ratings(path=PATH_RATINGS):
        ratings = pd.read_csv(path)
        ratings = ratings.rename(columns={'userId':'user', 'movieId':ID})
        ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
        return ratings

# Funció per convertir la cadena JSON en una llista de diccionaris
def convert_to_dict_list(string):
        return ast.literal_eval(string)

def credits(path, path_movies=PATH_MOVIES):
        movies = movies_metadata(path_movies)
        credits = pd.read_csv(path)

        id_movies = set(movies[ID])

        credits = credits[credits[ID].isin(id_movies)]

        credits['cast'] = credits['cast'].apply(convert_to_dict_list)
        credits['crew'] = credits['crew'].apply(convert_to_dict_list)

        cast_rows = []
        crew_rows = []

        for _,row in credits.iterrows():
                for cast_member in row.cast:
                        cast_member['movie_id'] = row[ID]
                        cast_rows.append(cast_member)
                
                for crew_member in row.crew:
                        crew_member['movie_id'] = row[ID]
                        crew_rows.append(crew_member)

        cast = pd.DataFrame(cast_rows)
        crew = pd.DataFrame(crew_rows)
        
        return cast.convert_dtypes(), crew.convert_dtypes()

def keywords(path=PATH_KEYWORDS):
        keywords = pd.read_csv(path)
        keywords['keywords'] = keywords['keywords'].apply(convert_to_dict_list)
        keywords['keywords_id'] = keywords['keywords'].apply(lambda x: [d['id'] for d in x])
        keywords['keywords'] = keywords['keywords'].apply(lambda x: [d['name'] for d in x])
        return keywords

def links(path=PATH_LINKS):
        return pd.read_csv(path)

def small_ratings():
        ratings = data_ratings(PATH_RATINGS_SMALL)
        movies = movies_metadata(PATH_MOVIES)
        movies = movies[movies[ID].isin(ratings[ID])]
        return ratings, movies



if __name__ == '__main__':
        # cast, crew = credits(PATH_CREDITS, PATH_MOVIES)
        # keywords = keywords(PATH_KEYWORDS)
        # links = links(PATH_LINKS)
        # links_small = links(PATH_LINKS_SMALL)
        # movies = movies_metadata(PATH_MOVIES)
        # ratings = data_ratings(PATH_RATINGS)
        ratings_small = small_ratings()

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