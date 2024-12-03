from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

class Evaluation:
    def __init__(self, true_ratings, predicted_ratings):
        self.__true_ratings = true_ratings
        self.__predicted_ratings = predicted_ratings

    def __extract_ratings(self):
        '''
        Separa el ground truth (y_true) i la predicció del nostra model.
        Transforma les y's en arrays.
        '''
        y_true = []
        y_pred = []
        for user_id in self.__true_ratings:
            for movie_id, true_rating in self.__true_ratings[user_id].items():
                if (user_id in self.__predicted_ratings) and (movie_id in self.__predicted_ratings[user_id]):
                    y_true.append(true_rating)
                    y_pred.append(self.__predicted_ratings[user_id][movie_id])
            return np.array(y_true), np.array(y_pred)

    def calculate_mae(self):
        '''
        Mean Absolute Error
        '''
        y_true