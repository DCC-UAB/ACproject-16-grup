import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from ast import literal_eval
import itertools

ratings = pd.read_csv('./datasets/ratings.csv')
movies = pd.read_csv("./datasets/movies_metadata.csv")
credits = pd.read_csv("./datasets/credits.csv")
keywords = pd.read_csv("./datasets/keywords.csv")

# Funció per mostrar la distribució de les puntuacions
def mostra_distribucio_ratings(ratings):
    plt.figure(figsize=(10, 6))
    ratings['rating'].hist(bins=20, edgecolor='black')
    plt.title('Distribució de Puntuacions')
    plt.xlabel('Puntuació')
    plt.ylabel('Nombre de Vots')
    plt.grid(False)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.show()

#mostra_distribucio_ratings(ratings)

# Funció per a analitzar les mètriques de les pel·lícules
def analitza_metrics_pellicules(movies, min_vots):
    print(f"Total de pel·lícules: {movies.shape[0]}\n")
    print(f"Valors nulls a les pel·lícules:\n{movies.isnull().sum()}\n")
    print(f"Pel·lícules més votades:\n{movies.sort_values('vote_count', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")
    print(f"Pel·lícules amb millor puntuació i més de {min_vots} vots:\n{movies[movies['vote_count'] > min_vots].sort_values('vote_average', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n")

#analitza_metrics_pellicules(movies, 1000)

# Funció per mostrar les pel·lícules més valorades segons la puntuació ponderada
def mostra_pellicules_mes_valorades(movies, C, m):
    movies["puntuacio_ponderada"] = (movies['vote_count']*movies['vote_average'] + C*m) / (movies['vote_count'] + m)
    top20 = movies.sort_values('puntuacio_ponderada', ascending=False).drop_duplicates('title').head(20)
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

C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.3)
#mostra_pellicules_mes_valorades(movies, C, m)

# Funció per analitzar els gèneres de les pel·lícules
def analitza_generes(movies):
    movies['genres'] = movies['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
    tots_generes = movies['genres'].explode()
    generes = tots_generes.value_counts().reset_index()
    generes.columns = ['gènere', 'comptador']
    plt.figure(figsize=(10, 6))
    sns.barplot(x='comptador', y='gènere', data=generes, palette='viridis')
    plt.title('Distribució de Gèneres')
    plt.xlabel('Nombre de Pel·lícules')
    plt.ylabel('Gènere')
    plt.show()

#analitza_generes(movies)

# Funció per comptar pel·lícules per intervals de durada
def compta_pellicules_per_durada(movies):
    mv = movies.dropna(subset=['runtime', 'popularity'])
    mv['popularity'] = mv['popularity'].astype(float)
    intervals = [0, 30, 80, 130, np.inf]
    etiquetes = ['Curmetratge', 'Pel·lícula curta', 'Pel·lícula', 'Pel·lícula llarga']
    runtime = pd.to_numeric(mv['runtime'], errors='coerce')
    durada = pd.cut(runtime, bins=intervals, labels=etiquetes, right=False)
    comptador_intervals = durada.value_counts().sort_index()
    popularitat_intervals = mv.groupby(durada)['popularity'].mean()
    popularitat_intervals = mv.groupby(durada)['popularity'].mean()
    taula_intervals = pd.DataFrame({'Comptador': comptador_intervals,'Popularitat Mitjana': popularitat_intervals})
    print(taula_intervals)

#compta_pellicules_per_durada(movies)

# Funció per mirar la relació entre la puntuació i el nombre de vots
def vote_vs_rate(movies):
    correlacio = movies[['vote_count', 'vote_average']].corr().iloc[0, 1]
    print(f"Correlació entre Nombre de Vots i Puntuació: {correlacio:.2f}")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='vote_count', y='vote_average', data=movies, alpha=0.7)
    plt.title('Puntuació vs Nombre de Vots')
    plt.xlabel('Nombre de Vots')
    plt.ylabel('Puntuació')
    plt.show()

#vote_vs_rate(movies)


# Funció per analitzar si els directors i actors més populars influeixen en la popularitat de les pel·lícules 
def analitza_directors_actors(credits, movies):
    crew = credits['crew'].apply(literal_eval)
    cast = credits['cast'].apply(literal_eval)

    credits['directors'] = crew.apply(lambda x: [i['name'] for i in x if i['job'] == 'Director'])
    directors = credits.explode('directors')['directors'].value_counts().reset_index()
    directors.columns = ['Director', 'Nombre de Pel·lícules']

    credits['actors'] = cast.apply(lambda x: [i['name'] for i in x])
    actors = credits.explode('actors')['actors'].value_counts().reset_index()
    actors.columns = ['Actor', 'Nombre de Pel·lícules']    

    print(f"Actors més influents:\n{actors.head(10)}\n")
    print(f"Directors més influents:\n{directors.head(10)}\n")
    #Falta mirar si aquests apareixen a els millors pel·lícules

#analitza_directors_actors(credits, movies)