import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

class ItemItemRecommender:
    def __init__(self):
        """
        Inicialitza el sistema de recomanació basat en ítems amb les dades de valoracions i ítems.
        """
        self.__ratings = None
        self.__ratings_matrix = None
        self.__item_similarity = None

    def load_data(self, file_path):
        """
        Carrega les dades de valoracions des d'un fitxer CSV.
        """
        try:
            self.__ratings = pd.read_csv(file_path)
            if self.__ratings.isnull().values.any():
                print("Atenció: Hi ha valors nuls a les dades carregades.")

            # Creem la matriu de ratings (usuaris x ítems) com un DataFrame
            self.__ratings_matrix = self.__ratings.pivot_table(index='userId', columns='movieId', values='rating')

            # Omplim NaN amb 0 (necessari per a similitud de cosinus)
            self.__ratings_matrix = self.__ratings_matrix.fillna(0)

        except Exception as e:
            print(f"Error carregant el fitxer: {e}")

    @property
    def item_similarity(self):
        return self.__item_similarity

    def similarity_matrix(self, method='cosine'):
        """
        Crea una matriu de similitud entre ítems utilitzant el mètode especificat.
        """
        if method == 'cosine':
            # Similitud de cosinus
            similarity_matrix = cosine_similarity(self.__ratings_matrix.T)
            self.__item_similarity = pd.DataFrame(
                similarity_matrix, 
                index=self.__ratings_matrix.columns, 
                columns=self.__ratings_matrix.columns
            )
        elif method == 'pearson':
            # Similitud de Pearson com una correlació
            self.__item_similarity = self.__ratings_matrix.corr(method='pearson')
        else:
            raise ValueError("Mètode desconegut: només 'cosine' o 'pearson' són vàlids.")

    def get_similar_items(self, item_id, top_n=5, method='cosine'):
        """
        Retorna els ítems més similars a un ítem donat.
        :param item_id: ID de l'ítem per al qual es volen recomanacions
        :param top_n: Nombre d'ítems similars que es volen obtenir
        :param method: Mètode de similitud ('cosine' o 'pearson')
        :return: DataFrame amb els ítems més similars
        """
        self.similarity_matrix(method)

        if item_id not in self.__item_similarity.columns:
            print(f"Warning: Ítem {item_id} no trobat a la matriu de similitud.")
            return pd.Series(dtype=float)  # Retorna un Series buit

        similarity_scores = self.__item_similarity[item_id]
        similar_items = similarity_scores.sort_values(ascending=False).head(top_n + 1)  # Incloent-se a si mateix
        #check if item_id in similar_items
        if item_id in similar_items:
            return similar_items.drop(item_id)  # Excloem el propi ítem
        return similar_items

    def recommend_for_user(self, user_id, top_n=5, method='cosine'):
        """
        Recomana ítems a un usuari basat en els ítems que ha valorat més altament.
        :param user_id: ID de l'usuari
        :param top_n: Nombre de recomanacions
        :param method: Mètode de similitud ('cosine' o 'pearson')
        :return: DataFrame amb les recomanacions per a l'usuari
        """
        if user_id not in self.__ratings_matrix.index:
            print(f"Warning: Usuari {user_id} no trobat a les dades.")
            return pd.Series(dtype=float)

        user_ratings = self.__ratings_matrix.loc[user_id]
        recommendations = {}

        for item_id, rating in user_ratings[user_ratings > 0].items():
            similar_items = self.get_similar_items(item_id, top_n=top_n, method=method)

            for similar_item, similarity in similar_items.items():
                if similar_item not in recommendations:
                    recommendations[similar_item] = 0

                recommendations[similar_item] += similarity * rating

        # Ordenem les recomanacions per puntuació
        recommended_items = pd.Series(recommendations).sort_values(ascending=False).head(top_n)
        return recommended_items

if __name__ == "__main__":
    recommender = ItemItemRecommender()
    recommender.load_data('./datasets/ratings_small.csv')

    user_id = 402
    item_id = 13

    # Recomanacions per a un usuari
    recommendations = recommender.recommend_for_user(user_id, top_n=5, method='cosine')
    print(f"Recomanacions per a l'usuari {user_id} (cosinus):\n{recommendations}")

    # Similitud per a un ítem
    similar_items = recommender.get_similar_items(item_id, top_n=5, method='cosine')
    print(f"\nÍtems similars a l'ítem {item_id} (cosinus):\n{similar_items}")