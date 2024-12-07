import sys
sys.path.append('./systems')
from user_to_user import UserUserRecommender
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

class Evaluation:
    def __init__(self, recommender, true_ratings):
        self.__recomender = recommender
        self.__true_ratings = true_ratings
        self.__predicted_ratings = None

    def generate_predictions(self, method='cosine', topN=20):
        """
        Genera les prediccions per als ratings no valorats.
        """
        self.__predicted_ratings = self.__recomender.generate_recommendation_matrix(method=method, topN=topN)
        return self.__predicted_ratings

    def __extract_ratings(self):
        '''
        Separa el ground truth (y_true) i la predicció del nostra model.
        Transforma les y's en arrays.
        '''

        y_true = []
        y_pred = []
        for user_id in self.__true_ratings:
            for movie_id, true_rating in self.__true_ratings[user_id].items():
                if (user_id in self.__predicted_ratings):
                    if (movie_id in self.__predicted_ratings[user_id]):
                        print(f"Coincidència trobada: user_id={user_id}, movie_id={movie_id}")
                        y_true.append(true_rating)
                        y_pred.append(self.__predicted_ratings[user_id][movie_id])
                    else:
                        print(f"Movie ID {movie_id} no trobada per usuari {user_id} a predicted ratings")

                else:
                    print(f"User ID {user_id} no trobat a predicted ratings")
        #print(f"y_true: {y_true}")
        #print(f"y_pred: {y_pred}")
        return np.array(y_true), np.array(y_pred)

    def calculate_mae(self):
        '''
        Mean Absolute Error
        '''
        y_true, y_pred = self.__extract_ratings()
        if len(y_true) == 0 or len(y_pred) == 0:
            print("No es pot calcular la MAE si les llistes estan buides")
            return None
        return mean_absolute_error(y_true, y_pred)
    

    def calculate_rsme(self):
        '''
        Root Mean Squared Error
        '''
        y_true, y_pred = self.__extract_ratings()
        if len(y_true) == 0 or len(y_pred) == 0:
            print("No es pot calcular la RSME si les llistes estan buides")
            return None
        return np.sqrt(mean_squared_error(np.log1p(y_true), np.log1p(y_pred)))
    

    def calculate_topn_precision(self, n = 10):
        '''
        TopN-Precisió. Per defecte, 10.
        '''
        precision_values = []
        for user_id in self.__true_ratings:
            #n millors items reals
            true_items = sorted(self.__true_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:n]
            
            #n millors items predits pel nostre model
            predicted_items = sorted(self.__predicted_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:n]

            #Per poder comparar, llistem només els IDs de les pel·lícules
            true_items_ids = []
            for item in true_items:
                true_items_ids.append(item[0])

            predicted_items_ids = []
            for item in predicted_items:
                predicted_items_ids.append(item[0])

            #Comprovem quants ítems predits estan també en els reals
            counter = 0
            for predicted_item in predicted_items_ids:
                if predicted_item in true_items_ids:
                    counter += 1

            #Calculem la precisió a través de la fórmula
            precision = counter / len(predicted_items_ids) if len(predicted_items_ids) > 0 else 0
            precision_values.append(precision)

        return np.mean(precision_values)
    

    def calculate_topn_recall(self, n=10):
        '''
        TopN-Recall. Per defecte, 10.
        '''
        recall_values = []
        for user_id in self.__true_ratings:
            # n millors ítems reals
            true_items = sorted(self.__true_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:n]

            # n millors ítems predits pel nostre model
            predicted_items = sorted(self.__predicted_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:n]

            # Per poder comparar, llistem només els IDs de les pel·lícules
            true_items_ids = []
            for item in true_items:
                true_items_ids.append(item[0])

            predicted_items_ids = []
            for item in predicted_items:
                predicted_items_ids.append(item[0])

            # Comprovem quants ítems predits estan també en els reals
            counter = 0
            for predicted_item in predicted_items_ids:
                if predicted_item in true_items_ids:
                    counter += 1

            # Calculem el recall a través de la fórmula
            recall = counter / len(true_items_ids) if len(true_items_ids) > 0 else 0
            recall_values.append(recall)

        return np.mean(recall_values)
    

    def calculate_ndcg(self, n=10):
        """
        Calcula el NDCG (Normalized Discounted Cumulative Gain).
        """
        ndcg_scores = []
        for user_id in self.__true_ratings:
            true_items = sorted(self.__true_ratings[user_id].items(), key=lambda x: x[1], reverse=True)
            predicted_items = sorted(self.__predicted_ratings[user_id].items(), key=lambda x: x[1], reverse=True)[:n]

            # Calcular DCG
            dcg = sum([true_items[i][1] / np.log2(i + 2) for i in range(len(predicted_items)) if i < len(true_items)])

            # Calcular IDCG (ideal DCG)
            idcg = sum([true_items[i][1] / np.log2(i + 2) for i in range(min(len(true_items), n))])

            ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
        return np.mean(ndcg_scores)

    def evaluate(self, method='cosine', topN=20):
        """
        Avaluem el sistema de recomanació amb diverses mètriques.
        """
        self.generate_predictions(method=method, topN=topN)
        mae = self.calculate_mae()
        rsme = self.calculate_rsme()
        topn_precision = self.calculate_topn_precision(n=topN)
        topn_recall = self.calculate_topn_recall(n=topN)
        ndcg = self.calculate_ndcg()
        
        return {
            'MAE': mae,
            'RSME': rsme,
            'Top-N Precision': topn_precision,
            'Top-N Recall': topn_recall,
            'NDCG': ndcg
        }


if __name__ == "__main__":
    ratings_df = pd.read_csv('./datasets/ratings_small.csv')

    recommender = UserUserRecommender()

    ratings_df = pd.read_csv('./datasets/ratings_small.csv')

    ground_truth = {}
    for _, row in ratings_df.iterrows():
        user_id = row['userId']
        movie_id = row['movieId']
        rating = row['rating']
        
        if user_id not in ground_truth:
            ground_truth[user_id] = {}
        
        ground_truth[user_id][movie_id] = rating

    # Inicialitzar l'evaluador
    evaluator = Evaluation(recommender, ground_truth)

    # Generar les prediccions
    predictions = evaluator.generate_predictions(method='cosine', topN=10)

    # Avaluar el sistema
    evaluation_metrics = evaluator.evaluate(method='cosine', topN=10)

    print(evaluation_metrics)