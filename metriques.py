import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from ast import literal_eval

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
