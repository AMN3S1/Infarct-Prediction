
import pandas as pd

file_path = 'heart_attack_prediction_indonesia.csv'

def load_data(file_path):

    df = pd.read_csv(file_path)
    return df

def basic_info(df):

    print("Size of dataset:", df.shape)
    print("First 5 rows:")
    print(df.head())
    print("Info about columns:")
    print(df.info())
    print("Nulls in columns:")
    print(df.isnull().sum())

if __name__ == '__main__':
    df = load_data(file_path)
    basic_info(df)
