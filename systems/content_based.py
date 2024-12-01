import sys
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

from data_preprocessing.preprocessing_csv import movies_metadata, ratings, credits
import pandas as pd

movies = movies_metadata('./datasets/movies_metadata.csv')
rates = ratings('./datasets/ratings.csv')
cast, credit = credits('./datasets/credits.csv')

print(movies.head(3))
print(rates.head(3))
print(credit.head(3))
