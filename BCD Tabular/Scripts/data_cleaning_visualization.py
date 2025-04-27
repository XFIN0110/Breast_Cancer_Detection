import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the correct dataset directory
csv_folder = "/home/xfin/Downloads/Major P/Dataset/csv/"

# List of dataset files
files = [
    "calc_case_description_train_set.csv",
    "calc_case_description_test_set.csv",
    "mass_case_description_train_set.csv",
    "mass_case_description_test_set.csv"
]

# Check if the directory exists
if not os.path.exists(csv_folder):
    raise FileNotFoundError(f"Error: Directory not found -> {csv_folder}")

# Check if all files exist in the directory
missing_files = [file for file in files if not os.path.exists(os.path.join(csv_folder, file))]
if missing_files:
    raise FileNotFoundError(f"Error: The following files are missing in {csv_folder}: {missing_files}")

# Load datasets
datasets = {file: pd.read_csv(os.path.join(csv_folder, file)) for file in files}

# Fill missing values with "Unknown" for categorical & mode for numerical
for name, df in datasets.items():
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype == "object":
                df.loc[:, col] = df[col].fillna("Unknown")  # Fixed chained assignment warning
            else:
                df.loc[:, col] = df[col].fillna(df[col].mode()[0])  # Fixed chained assignment warning

# Merge train sets for visualization
df_train = pd.concat([
    datasets["calc_case_description_train_set.csv"], 
    datasets["mass_case_description_train_set.csv"]
])

# Breast Density Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="breast density", hue="breast density", data=df_train, palette="coolwarm", legend=False)  # Fixed warning
plt.title("Breast Density Distribution")
plt.xlabel("Breast Density")
plt.ylabel("Count")
plt.show()

# Pathology Class Balance (Benign vs Malignant)
plt.figure(figsize=(6, 4))
sns.countplot(x="pathology", hue="pathology", data=df_train, palette="coolwarm", legend=False)  # Fixed warning
plt.title("Pathology Distribution (Benign vs Malignant)")
plt.xlabel("Pathology")
plt.ylabel("Count")
plt.show()

# Histogram of Assessment Scores
plt.figure(figsize=(6, 4))
sns.histplot(df_train["assessment"], bins=10, kde=True, color="blue")
plt.title("Assessment Score Distribution")
plt.xlabel("Assessment Score")
plt.ylabel("Frequency")
plt.show()

# Abnormality Type Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x="abnormality type", hue="abnormality type", data=df_train, palette="coolwarm", legend=False)  # Fixed warning
plt.title("Abnormality Type Distribution")
plt.xlabel("Abnormality Type")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

