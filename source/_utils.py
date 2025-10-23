# src/_utils.py
import pandas as pd
from .preprocess import preprocess  # relative import

def load_data_and_preprocess():
    # Load CSV relative to project root
    df = pd.read_csv('data/adult.csv')
    return preprocess(df)
