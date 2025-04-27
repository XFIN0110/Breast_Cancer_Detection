import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the cleaned datasets
dataset_path = "/home/xfin/Downloads/Major P/Dataset/csv/"
files = ["calc_case_description_train_set.csv", "mass_case_description_train_set.csv"]

# Read and merge train datasets
df_train = pd.concat([pd.read_csv(dataset_path + file) for file in files])

# Drop unnecessary columns (if any)
df_train.drop(columns=["image file path"], errors="ignore", inplace=True)

# Encode categorical columns
categorical_cols = ["breast density", "abnormality type", "subtlety", "pathology"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_train[col] = le.fit_transform(df_train[col])
    label_encoders[col] = le  # Store the encoder for future use

# Normalize numerical features
scaler = StandardScaler()
df_train["assessment"] = scaler.fit_transform(df_train[["assessment"]])

# Split data into features (X) and target (y)
X = df_train.drop(columns=["pathology"])
y = df_train["pathology"]

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Save processed data
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

print("Data preprocessing completed. Processed files saved.")

