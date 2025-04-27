AI-Powered Breast Cancer Detection Using Image Segmentation
Project Overview

This project aims to develop an AI-powered system for breast cancer detection using both tabular data and medical image segmentation. It assists healthcare professionals by automating early detection and diagnosis through machine learning models and lightweight CNNs.
Project Structure

├── BCD_Tabular        
|---Dataset/                     # Tabular data files (CSV, Excel)
├── models/                      # Saved trained models (.pkl, .h5)
│   └── breast_cancer_model.pkl   # Random Forest trained model
├── scripts/                     # Python scripts
│   ├── data_cleaning_visualization.py
│   ├── data_preprocessing.py
│   └── model_training.py
├── README.md                     # Project overview and instructions


Install required dependencies:



Prepare the Dataset:

    Place your tabular dataset (.csv files) inside the BCD_Tabular/ folder.

    If using images, create a data/ folder separately (instructions inside the corresponding script if needed).

Run the scripts sequentially:

    Data cleaning and visualization:

python scripts/data_cleaning_visualization.py

Data preprocessing:

python scripts/data_preprocessing.py

Model training:

        python scripts/model_training.py

Features

    Data Cleaning and Exploratory Data Analysis (EDA)

    Feature Engineering and Label Encoding

    Random Forest Classifier for tabular dataset

    Lightweight CNN-based image classification (planned)

    Basic OpenCV image segmentation (planned)

Dataset

Important:
Due to size and licensing restrictions, datasets are not included in this repository.

    Please manually place your .csv files inside the /BCD_Tabular/ folder.

Example CSV:

    breast_cancer_data.csv

    processed_cancer_data.csv

Future Enhancements

    Implement lightweight CNN for medical image classification.

    Basic segmentation with OpenCV.

    Hyperparameter tuning and optimization.

    Deployment through Flask/Django web application.

