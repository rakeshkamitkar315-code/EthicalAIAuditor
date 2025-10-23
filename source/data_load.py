import pandas as pd

def load_data(path='data/adult.csv'):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    df = load_data()
    print("Rows:", len(df))
    print(df.columns)
    print(df.head())