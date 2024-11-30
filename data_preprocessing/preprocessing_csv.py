import pandas as pd
import ast

def movies_metadata(path):
        items = pd.read_csv(path)
        items['id'] = pd.to_numeric(items['id'], errors='coerce')
        items = items.dropna(subset=['id'])
        items['id'] = items['id'].astype('int64')
        items['adult'] = items['adult'].map({'True': True, 'False': False})
        items['adult']= items['adult'].astype('bool')
        items = items.convert_dtypes()
        return items
    
def ratings(path):
        ratings = pd.read_csv(path)
        ratings = ratings.rename(columns={'userId':'user', 'movieId':'id'})
        ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
        return ratings

# Funció per convertir la cadena JSON en una llista de diccionaris
def convert_to_dict_list(string):
        return ast.literal_eval(string)


# TODO: Implementar la funció credits, on es desglossaran les dades de la columna 'cast' i 'crew' del fitxer 'credits.csv'
def credits(path, path_movies='./datasets/movies_metadata.csv'):
        movies = movies_metadata(path_movies)
        credits = pd.read_csv(path)
        # credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
        # credits = credits.dropna(subset=['id'])
        # credits['id'] = credits['id'].astype('int64')
        # credits = credits.convert_dtypes()
        id_movies = set(movies['id'])
        credits = credits[credits['id'].isin(id_movies)]

        # Convertir la columna 'cast' en llistes de diccionaris
        credits['cast'] = credits['cast'].apply(convert_to_dict_list)

        # Crear un nou DataFrame per emmagatzemar les files desglossades
        cast = pd.DataFrame()

        # Iterar sobre cada línia del DataFrame original
        for index, row in credits.iterrows():
                # Obtenir la llista de diccionaris per a la columna 'cast'
                cast_list = row['cast']
                
                # Afegir l'ID de la pel·lícula a cada element de la llista 'cast'
                for cast_member in cast_list:
                        cast_member['movie_id'] = row['id']  # Afegir l'ID original de la pel·lícula
                        
                # Afegir els diccionaris de la llista 'cast' com noves files al nou DataFrame
                cast = pd.concat([cast, pd.DataFrame(cast_list)], ignore_index=True)

        # Mostrar el nou DataFrame
        print(cast)

        
        return cast, credits
