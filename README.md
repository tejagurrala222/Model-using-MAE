# Q1 - Used Car Data Cleaning + Baseline MAE
# ---------------------------------------------------
# Replace file name below with your dataset path

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Load dataset
df = pd.read_csv("used_cars.csv")

# ===================================================
# Task 1 — Explore and Identify Issues
# ===================================================

print("Shape of dataset:")
print(df.shape)

print("\nInfo:")
print(df.info())

print("\nDescribe:")
print(df.describe(include='all'))

# Example issues observed:
# 1. Null values in selling_price (target)
# 2. Missing values in input columns
# 3. brand column has inconsistent casing/spaces
# 4. mileage stored as text like '18 kmpl'
# 5. Duplicate rows may exist

# ===================================================
# Task 2 — Clean the Data
# ===================================================

# 1. Drop rows where target is null
df = df.dropna(subset=["selling_price"])

# 2. Clean brand column
if "brand" in df.columns:
    df["brand"] = df["brand"].astype(str).str.strip().str.lower()

# 3. Extract numeric values from mileage column
if "mileage" in df.columns:
    df["mileage"] = df["mileage"].astype(str).str.extract(r'(\d+\.?\d*)')
    df["mileage"] = pd.to_numeric(df["mileage"], errors="coerce")

# 4. Impute missing input features
for col in df.columns:
    if col != "selling_price":
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# 5. Remove duplicates
df = df.drop_duplicates()

print("\nCleaned Shape:")
print(df.shape)

# ===================================================
# Task 3 — Baseline MAE
# ===================================================

# Predict mean selling_price for all rows
mean_price = df["selling_price"].mean()

y_true = df["selling_price"]
y_pred = np.full(len(df), mean_price)

mae = mean_absolute_error(y_true, y_pred)

print("\nBaseline Mean Prediction:", mean_price)
print("Baseline MAE:", mae)
