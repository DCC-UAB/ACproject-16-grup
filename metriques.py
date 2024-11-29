import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from ast import literal_eval
import itertools

class preprocessing:
    def __init__(self):
        self.ratings = pd.read_csv('./datasets/ratings.csv')
        self.movies = pd.read_csv("./datasets/movies_metadata.csv")#, dtype={'id':'int64'})
        self.credits = pd.read_csv("./datasets/credits.csv")
        self.keywords = pd.read_csv("./datasets/keywords.csv")
        # self.links = pd.read_csv("./datasets/links.csv")

    # Funció per mostrar la distribució de les puntuacions    
    def mostra_distribucio_ratings(self):
        plt.figure(figsize=(10, 6))
        self.ratings['rating'].hist(bins=20, edgecolor='black')
        plt.title('Distribució de Puntuacions')
        plt.xlabel('Puntuació')
        plt.ylabel('Nombre de Vots')
        plt.grid(False)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
        plt.show()

    # Funció per a analitzar les mètriques de les pel·lícules
    def analitza_metrics_pellicules(self, min_vots):
        print(f"Total de pel·lícules: {self.movies.shape[0]}\n")
        print(f"Valors nulls a les pel·lícules:\n{self.movies.isnull().sum()}\n")
        print(f"Pel·lícules més votades:\n{self.movies.sort_values('vote_count', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")
        print(f"Pel·lícules amb millor puntuació i més de {min_vots} vots:\n{self.movies[self.movies['vote_count'] > min_vots].sort_values('vote_average', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")

    # Funció per mostrar les pel·lícules més valorades segons la puntuació ponderada
    def mostra_pellicules_mes_valorades(self, C, m):
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
    def compta_pellicules_per_durada(self):
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
    def analitza_directors_actors(self):
        crew = self.credits['crew'].apply(literal_eval)
        cast = self.credits['cast'].apply(literal_eval)

        self.credits['directors'] = crew.apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])
        directors = self.credits.explode('directors')['directors'].value_counts().reset_index()
        directors.columns = ['Director', 'Nombre de Pel·lícules']

        self.credits['actors'] = cast.apply(lambda x: [i['name'] for i in x])
        actors = self.credits.explode('actors')['actors'].value_counts().reset_index()
        actors.columns = ['Actor', 'Nombre de Pel·lícules']    

        print(f"Actors més influents:\n{actors.head(10)}\n")
        print(f"Directors més influents:\n{directors.head(10)}\n")
        
        #Falta mirar si aquests apareixen a els millors pel·lícules
    

if __name__=="__main__":
    p = preprocessing()
    p.mostra_distribucio_ratings()
    p.analitza_metrics_pellicules(1000) 
    
    C = p.movies['vote_average'].mean()
    m = p.movies['vote_count'].quantile(0.3)
    p.mostra_pellicules_mes_valorades(C, m)
    p.analitza_generes()
    p.compta_pellicules_per_durada()
    p.vote_vs_rate()
    p.analitza_directors_actors()


