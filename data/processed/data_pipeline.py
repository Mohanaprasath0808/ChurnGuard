import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset
df = pd.read_csv('updated customerchurn.csv')

# --- 1. SCHEMA & DATA DICTIONARY ---
print("================ CHURN DATA SCHEMA & DICTIONARY ================")
schema_df = pd.DataFrame({
    'Column_Name': df.columns,
    'Data_Type': df.dtypes.values,
    'Non_Null_Count': df.notnull().sum().values,
    'Null_Count': df.isnull().sum().values
})
print(schema_df)

# --- 2. DATA VALIDATION ---
print("\n================ DATA VALIDATION CHECKS ================")
print("Duplicate rows count:", df.duplicated().sum())
print("Missing values per column:\n", df.isnull().sum())

# Value validation (e.g., negative charges or invalid tenure)
print("Negative monthly charges count:", (df['monthly_charges'] < 0).sum())
print("Invalid tenure months (<0):", (df['tenure_months'] < 0).sum())

# --- 3. EDA (EXPLORATORY DATA ANALYSIS) ---
print("\n================ EDA SUMMARY STATISTICS ================")
print(df.describe())

# Target variable distribution
print("\nChurned 90d Distribution:")
print(df['churned_90d'].value_counts(normalize=True) * 100)

# --- 4. DATA LEAKAGE CHECKS ---
print("\n================ DATA LEAKAGE CHECKS ================")
# Check correlation of numeric columns with the target variable 
# (An unusually high correlation > 0.95 might indicate data leakage)
numeric_cols = df.select_dtypes(include=[np.number])
correlations = numeric_cols.corr()['churned_90d'].sort_values(ascending=False)
print("Correlation with target (churned_90d):\n", correlations)
