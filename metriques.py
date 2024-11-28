import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker


ratings = pd.read_csv('./datasets/ratings.csv')
movies = pd.read_csv("./datasets/movies_metadata.csv")
credits = pd.read_csv("./datasets/credits.csv")
keywords = pd.read_csv("./datasets/keywords.csv")


def show_ratings_distribution(ratings):
    plt.figure(figsize=(10, 6))
    ratings['rating'].hist(bins=20, edgecolor='black')
    plt.title('Distribució de Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Nombre de Vots')
    plt.grid(False)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.show()

#show_ratings_distribution(ratings)


def movies_metrics(movies, min_votes):
    print(f"Total de pel·lícules: {movies.shape[0]}\n\n")
    print(f"Valors nulls a les pel·lícules:\n{movies.isnull().sum()}\n\n")
    print(f"Pel·lícules mes votades:\n{movies.sort_values('vote_count', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n\n")
    print(f"Pel·lícules amb millor puntuació amb més de {min_votes} vots:\n{movies[movies['vote_count'] > min_votes].sort_values('vote_average', ascending=False)[['id','title', 'vote_count', 'vote_average']].head(10)}\n\n")
    
#movies_metrics(movies, 1000)

def show_movies_rates(movies,  C, m):
    movies["w_ratings"]= (movies['vote_count']*movies['vote_average'] + C*m) / (movies['vote_count'] + m)
    top20 = movies.sort_values('w_ratings', ascending=False).drop_duplicates('title').head(20)
    sns.barplot(x='w_ratings', y='title', data=top20, palette='viridis')
    plt.title('Top 20 Movies by Weighted Rating',fontsize=16)
    plt.xlabel('Weighted Rating (WR)', fontsize=12)
    plt.ylabel('Movie Title', fontsize=12)
    min_wr = top20['w_ratings'].min()
    max_wr = top20['w_ratings'].max()
    plt.xlim(min_wr - 0.1, max_wr + 0.1)
    for index, value in enumerate(top20['w_ratings']):
        plt.text(value - 0.001, index, f'{value:.3f}', color='black', va="center")

    plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    plt.show()

C = movies['vote_average'].mean()
m = movies['vote_count'].quantile(0.3)
show_movies_rates(movies, m, C)
