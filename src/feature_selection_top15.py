import pandas as pd

# Load original preprocessed dataset
df = pd.read_csv("preprocessed_data.csv")

# Top 15 features from CatBoost importance
top_15_features = ['previous_heart_disease', 'hypertension', 'diabetes', 'smoking_status', 'obesity', 'age',
                   'cholesterol_level', 'fasting_blood_sugar', 'cholesterol_ldl', 'triglycerides', 'sleep_hours',
                   'blood_pressure_systolic', 'waist_circumference', 'blood_pressure_diastolic', 'cholesterol_hdl',
                   "heart_attack"]

# Add target variable

# Create new dataframe with selected features
df_top15 = df[top_15_features]

# Save to CSV
df_top15.to_csv("top15_features_dataset.csv", index=False)

print("Top 15 feature dataset saved as 'top15_features_dataset.csv'")
