import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data_preprocessing')))

from preprocessing_csv import movies_metadata, ratings
import pandas as pd


print(movies_metadata('./datasets/movies_metadata.csv').head(1))
print(ratings('./datasets/ratings.csv').head(1))
