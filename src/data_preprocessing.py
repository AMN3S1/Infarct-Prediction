
import pandas as pd
import numpy as np

# Path to the dataset
file_path = 'heart_attack_prediction_indonesia.csv'


def load_data(file_path):
    """Load the dataset from a CSV file and return a DataFrame."""
    df = pd.read_csv(file_path)
    return df


def preprocess_data(df):
    """
    Preprocess the data:
    - Convert date columns (if any) to datetime.
    - Fill missing numerical values with the median.
    - Fill missing categorical values (especially 'alcohol_consumption') with "Unknown".

    Returns:
        df: Preprocessed DataFrame.
        numeric_features: List of numeric feature names.
        categorical_features: List of categorical feature names.
    """
    # Convert 'date' column to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Define numeric features based on the dataset columns
    numeric_features = [
        'age',
        'cholesterol_level',
        'waist_circumference',
        'sleep_hours',
        'blood_pressure_systolic',
        'blood_pressure_diastolic',
        'fasting_blood_sugar',
        'cholesterol_hdl',
        'cholesterol_ldl',
        'triglycerides',
        'hypertension',
        'diabetes',
        'obesity',
        'family_history',
        'previous_heart_disease',
        'medication_usage',
        'participated_in_free_screening'
    ]

    # Define categorical features based on the dataset columns
    categorical_features = [
        'gender',
        'region',
        'income_level',
        'smoking_status',
        'alcohol_consumption',
        'physical_activity',
        'dietary_habits',
        'air_pollution_exposure',
        'stress_level',
        'EKG_results'
    ]

    # Fill missing numerical values with the median
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].median())

    # Fill missing values in 'alcohol_consumption' with "Unknown"
    if 'alcohol_consumption' in df.columns:
        df['alcohol_consumption'] = df['alcohol_consumption'].fillna("Unknown")

    # Fill missing values in other categorical features with mode
    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df, numeric_features, categorical_features


if __name__ == '__main__':
    df = load_data(file_path)
    df, num_feats, cat_feats = preprocess_data(df)
    print("Data after preprocessing:")
    print(df.head())
# Save the preprocessed DataFrame to a CSV file
    df.to_csv("preprocessed_data.csv", index=False)
    print("Preprocessed data has been saved to 'preprocessed_data.csv'.")