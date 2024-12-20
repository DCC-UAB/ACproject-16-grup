import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)
from data_preprocessing.preprocessing_csv import small_ratings

class preprocessing:
    def __init__(self):
        self.ratings, self.movies, self.credits, self.keywords = small_ratings()

    
    def numerical_analysis(self):
        """
        Mostra l'anàlisi bàsic de les dades 
        """
        print(f"\nNombre de pel·lícules: {self.movies.shape[0]}")
        print(f"Nombre d'users: {self.ratings['user'].nunique()}")
        print(f"Nombre de valoracions: {self.ratings.shape[0]}")
        sparsity = 1 - (self.ratings.shape[0] / (self.movies.shape[0] * self.ratings['user'].nunique()))
        print(f"\nSparsity del dataset: {sparsity:.2%}")

        print(f"\nNombre de valors nulls a les pelicules:")
        null_counts = self.movies.isnull().sum()
        for column, count in null_counts.items():
            print(f"\tCategoria: {column} - Nombre de valors nulls: {count}")


    def ratings_per_user_distribution(self):
        """
        Mostra la distribució del nombre de valoracions per usuari.
        """
        user_activity = self.ratings['user'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(user_activity, bins=50, kde=False, edgecolor="black")
        plt.title("Distribució del nombre de valoracions per usuari")
        plt.xlabel("Nombre de valoracions")
        plt.ylabel("Nombre d'usuaris")
        plt.show()

        
    def ratings_distribution(self):
        """
        Mostra la distribució de les puntuacions.
        """
        plt.figure(figsize=(10, 6))
        self.ratings["rating"].hist(bins=20, edgecolor="black")
        plt.title("Distribució de Puntuacions")
        plt.xlabel("Puntuació")
        plt.ylabel("Nombre de Vots")
        plt.grid(False)
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        plt.show()


    def ratings_per_movie_distribution(self):
        """
        Mostra la distribució del nombre de valoracions per pel·lícula.
        """
        movie_popularity = self.ratings['id'].value_counts()
        plt.figure(figsize=(10, 6))
        sns.histplot(movie_popularity, bins=50, kde=False, edgecolor="black")
        plt.title("Distribució del nombre de valoracions per pel·lícula")
        plt.xlabel("Nombre de valoracionss")
        plt.ylabel("Nombre de pel·lícules")
        plt.show()


    def top_rated_movies(self):
        """
        Mostra les 20 pel·lícules més ben valorades.
        """
        C = self.movies["vote_average"].mean()
        m = self.movies["vote_count"].quantile(0.3)
        self.movies["puntuacio_ponderada"] = (
            self.movies["vote_count"] * self.movies["vote_average"] + C * m
        ) / (self.movies["vote_count"] + m)
        top20 = (
            self.movies.sort_values("puntuacio_ponderada", ascending=False)
            .drop_duplicates("title")
            .head(20)
        )
        sns.barplot(x="puntuacio_ponderada", y="title", data=top20, palette="viridis")
        plt.title("Top 20 Pel·lícules per Puntuació Ponderada", fontsize=16)
        plt.xlabel("Puntuació Ponderada (PP)", fontsize=12)
        plt.ylabel("Títol de la Pel·lícula", fontsize=12)
        min_pp = top20["puntuacio_ponderada"].min()
        max_pp = top20["puntuacio_ponderada"].max()
        plt.xlim(min_pp - 0.1, max_pp + 0.1)
        for index, value in enumerate(top20["puntuacio_ponderada"]):
            plt.text(value - 0.001, index, f"{value:.3f}", color="black", va="center")

        plt.gca().xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
        plt.show()


    def analitza_generes(self):
        """
        Mostra la distribució de gèneres de les pel·lícules.
        """
        self.movies['genres'] = self.movies['genres'].apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])
        tots_generes = self.movies['genres'].explode()
        generes = tots_generes.value_counts().reset_index()
        generes.columns = ["gènere", "comptador"]
        plt.figure(figsize=(10, 6))
        sns.barplot(x="comptador", y="gènere", data=generes, palette="viridis")
        plt.title("Distribució de Gèneres")
        plt.xlabel("Nombre de Pel·lícules")
        plt.ylabel("Gènere")
        plt.show()


    def movies_runtime_distribution(self):
        """
        Mostra la distribució de la durada de les pel·lícules i la seva popularitat.
        """
        mv = self.movies.dropna(subset=["runtime", "popularity"])
        mv["popularity"] = mv["popularity"].astype(float)
        intervals = [0, 30, 80, 130, np.inf]
        etiquetes = [
            "Curtmetratge",
            "Pel·lícula curta",
            "Pel·lícula",
            "Pel·lícula llarga",
        ]
        runtime = pd.to_numeric(mv["runtime"], errors="coerce")
        durada = pd.cut(runtime, bins=intervals, labels=etiquetes, right=False)
        comptador_intervals = durada.value_counts().sort_index()
        popularitat_intervals = mv.groupby(durada)["popularity"].mean()
        popularitat_intervals = mv.groupby(durada)["popularity"].mean()
        taula_intervals = pd.DataFrame(
            {
                "Quantitat": comptador_intervals,
                "Popularitat Mitjana": popularitat_intervals,
            }
        )
        print(taula_intervals)


    def vote_vs_rate(self):
        """
        Mostra la correlació entre el nombre de vots i la puntuació de les pel·lícules.
        """
        correlacio = self.movies[["vote_count", "vote_average"]].corr().iloc[0, 1]
        print(f"\nCorrelació entre Nombre de Vots i Puntuació: {correlacio:.2f}")
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="vote_count", y="vote_average", data=self.movies, alpha=0.7)
        plt.title("Puntuació vs Nombre de Vots")
        plt.xlabel("Nombre de Vots")
        plt.ylabel("Puntuació")
        plt.show()
 

    def dataset_density(self, user_limit=500, movie_limit=500):
        """
        Mostra la densitat de les interaccions entre usuaris i pel·lícules.
        """
        top_users = self.ratings['user'].value_counts().head(user_limit).index
        top_movies = self.ratings['id'].value_counts().head(movie_limit).index
        subsample = self.ratings[self.ratings['user'].isin(top_users) & self.ratings['id'].isin(top_movies)]
        interaction_matrix = pd.pivot_table(subsample, values='rating', index='user', columns='id', fill_value=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(interaction_matrix, cmap="viridis", cbar=True, xticklabels=False, yticklabels=False)
        plt.title("Densitat d'interaccions")
        plt.xlabel("Pel·lícules")
        plt.ylabel("Usuaris")
        plt.show()



if __name__ == "__main__":
    p = preprocessing()
    p.numerical_analysis()
    p.ratings_per_user_distribution()
    p.ratings_distribution()
    p.ratings_per_movie_distribution()
    p.dataset_density()
    p.vote_vs_rate()
    p.top_rated_movies()
    p.analitza_generes()
    p.movies_runtime_distribution()
    


