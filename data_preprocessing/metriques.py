import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from ast import literal_eval
import itertools

import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import movies_metadata, ratings

class preprocessing:
    def __init__(self):
        self.ratings = ratings('./datasets/ratings.csv')
        self.credits = pd.read_csv("./datasets/credits.csv")
        self.movies = movies_metadata("./datasets/movies_metadata.csv")
        # self.preprocessing_data()
        self.movies = self.movies[self.movies['id'].isin(self.credits['id'])]
        # self.keywords = pd.read_csv("./datasets/keywords.csv")
        # self.links = pd.read_csv("./datasets/links.csv")

    # def preprocessing_data(self):
    #     self.movies['id'] = pd.to_numeric(self.movies['id'], errors='coerce')
    #     self.movies = self.movies.dropna(subset=['id'])
    #     self.movies['id'] = self.movies['id'].astype('int64')
    #     self.movies = self.movies.convert_dtypes()
    #     return self.movies
        
    # Funció per mostrar la distribució de les puntuacions    
    def display_rating_distribution(self):
        plt.figure(figsize=(10, 6))
        self.ratings['rating'].hist(bins=20, edgecolor='black')
        plt.title('Distribució de Puntuacions')
        plt.xlabel('Puntuació')
        plt.ylabel('Nombre de Vots')
        plt.grid(False)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.show()

    # Funció per a analitzar les mètriques de les pel·lícules
    def movie_metrics(self, min_vots):
        print(f"Total de pel·lícules: {self.movies.shape[0]}\n")
        print(f"Valors nulls a les pel·lícules:\n{self.movies.isnull().sum()}\n")
        print(f"Pel·lícules més votades:\n{self.movies.sort_values('vote_count', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")
        print(f"Pel·lícules amb millor puntuació i més de {min_vots} vots:\n{self.movies[self.movies['vote_count'] > min_vots].sort_values('vote_average', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")

    # Funció per mostrar les pel·lícules més valorades segons la puntuació ponderada
    def top_rated_movies(self):
        C = self.movies['vote_average'].mean()
        m = self.movies['vote_count'].quantile(0.3)
        self.movies["puntuacio_ponderada"] = (self.movies['vote_count']*self.movies['vote_average'] + C*m) / (self.movies['vote_count'] + m)
        top20 = self.movies.sort_values('puntuacio_ponderada', ascending=False).drop_duplicates('title').head(20)
        sns.barplot(x='puntuacio_ponderada', y='title', data=top20, palette='viridis')
        plt.title('Top 20 Pel·lícules per Puntuació Ponderada', fontsize=16)
        plt.xlabel('Puntuació Ponderada (PP)', fontsize=12)
        plt.ylabel('Títol de la Pel·lícula', fontsize=12)
        min_pp = top20['puntuacio_ponderada'].min()
        max_pp = top20['puntuacio_ponderada'].max()
        plt.xlim(min_pp - 0.1, max_pp + 0.1)
        for index, value in enumerate(top20['puntuacio_ponderada']):
            plt.text(value - 0.001, index, f'{value:.3f}', color='black', va="center")

        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        plt.show()
    
    # Funció per analitzar els gèneres de les pel·lícules
    def analitza_generes(self):
        self.movies['genres'] = self.movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        tots_generes = self.movies['genres'].explode()
        generes = tots_generes.value_counts().reset_index()
        generes.columns = ['gènere', 'comptador']
        plt.figure(figsize=(10, 6))
        sns.barplot(x='comptador', y='gènere', data=generes, palette='viridis')
        plt.title('Distribució de Gèneres')
        plt.xlabel('Nombre de Pel·lícules')
        plt.ylabel('Gènere')
        plt.show()

    # Funció per comptar pel·lícules per intervals de durada
    def movies_runtime_distribution(self):
        mv = self.movies.dropna(subset=['runtime', 'popularity'])
        mv['popularity'] = mv['popularity'].astype(float)
        intervals = [0, 30, 80, 130, np.inf]
        etiquetes = ['Curtmetratge', 'Pel·lícula curta', 'Pel·lícula', 'Pel·lícula llarga']
        runtime = pd.to_numeric(mv['runtime'], errors='coerce')
        durada = pd.cut(runtime, bins=intervals, labels=etiquetes, right=False)
        comptador_intervals = durada.value_counts().sort_index()
        popularitat_intervals = mv.groupby(durada)['popularity'].mean()
        popularitat_intervals = mv.groupby(durada)['popularity'].mean()
        taula_intervals = pd.DataFrame({'Quantitat': comptador_intervals,'Popularitat Mitjana': popularitat_intervals})
        print(taula_intervals)


    # Funció per mirar la relació entre la puntuació i el nombre de vots
    def vote_vs_rate(self):
        correlacio = self.movies[['vote_count', 'vote_average']].corr().iloc[0, 1]
        print(f"Correlació entre Nombre de Vots i Puntuació: {correlacio:.2f}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='vote_count', y='vote_average', data=self.movies, alpha=0.7)
        plt.title('Puntuació vs Nombre de Vots')
        plt.xlabel('Nombre de Vots')
        plt.ylabel('Puntuació')
        plt.show()

    # Funció per analitzar si els directors i actors més populars influeixen en la popularitat de les pel·lícules 
    def get_most_popular_cast_crew(self):
        crew = self.credits['crew'].apply(literal_eval)
        cast = self.credits['cast'].apply(literal_eval)

        self.credits['directors'] = crew.apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])
        directors = self.credits.explode('directors')[['id', 'directors']].dropna(subset=['directors'])

        directors_counts = directors['directors'].value_counts().reset_index()
        directors_counts.columns = ['Director', 'Nombre de Pel·lícules']

        directors_ids = directors.groupby('directors')['id'].apply(list).reset_index()
        directors_ids.columns = ['Director', 'id']

        directors_final = pd.merge(directors_counts, directors_ids, on='Director')
        top10_directors = directors_final.head(10)
        
        
        self.credits['actors'] = cast.apply(lambda x: [i['name'] for i in x])
        actors = self.credits.explode('actors')[['id', 'actors']].dropna(subset=['actors'])

        actors_counts = actors['actors'].value_counts().reset_index()
        actors_counts.columns = ['Actor', 'Nombre de Pel·lícules']

        actors_ids = actors.groupby('actors')['id'].apply(list).reset_index()
        actors_ids.columns = ['Actor', 'id']

        actors_final = pd.merge(actors_counts, actors_ids, on='Actor')
        top10_actors = actors_final.head(10)
        
        print(f"Actors més influents:\n{top10_actors[['Actor','Nombre de Pel·lícules']]}\n")
        print(f"Directors més influents:\n{top10_directors[['Director', 'Nombre de Pel·lícules']]}\n")
        
        dins_millors_pelis = {}
        top20 = self.movies.sort_values('puntuacio_ponderada', ascending=False).drop_duplicates('title').head(20)
        for (_, actor_row), (_, director_row) in zip(top10_actors.iterrows(), top10_directors.iterrows()):
            dins_millors_pelis[actor_row['Actor']] = ['actor', 0]
            dins_millors_pelis[director_row['Director']] = ['director', 0]
            for peli_actor in actor_row['id']:
                if peli_actor in top20['id'].values:  
                    dins_millors_pelis[actor_row['Actor']][1] += 1
            for peli_director in director_row['id']:
                if peli_director in top20['id'].values:
                    dins_millors_pelis[director_row['Director']][1] += 1
        
        data = {'Nom': [], 'Rol': [], 'Comptador': []}
        for nom, (rol, comptador) in dins_millors_pelis.items():
            data['Nom'].append(nom)
            data['Rol'].append(rol)
            data['Comptador'].append(comptador)

        df = pd.DataFrame(data)

        # Gràfic de barres
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Comptador', y='Nom', hue='Rol', data=df, palette='viridis')
        plt.title('Actors i Directors més repetits a les 20 pel·lícules més votades', fontsize=16)
        plt.xlabel('Nombre de Vegades a les 20 Millors Pel·lícules', fontsize=12)
        plt.ylabel('Nom', fontsize=12)
        plt.legend(title='Rol', loc='upper right')
        plt.show()

    
if __name__=="__main__":
    p = preprocessing()
    p.display_rating_distribution()
    p.movie_metrics(1000) 
    
    p.top_rated_movies()
    p.analitza_generes()
    p.movies_runtime_distribution()
    p.vote_vs_rate()
    p.get_most_popular_cast_crew()



