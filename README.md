## Cardiovascular Disease Prediction Using Machine Learning Models

### Introduction
Cardiovascular disease (CVD) is a leading cause of morbidity and mortality worldwide. Early prediction of CVD risk using patient data can help guide preventive care and treatment. 
This project aims to compare several machine learning models for predicting CVD based on clinical and lifestyle features, and to identify the most effective approach for this task.
This repository contains the implementation of multiple machine learning models to predict cardiovascular disease (CVD) based on patient data. 
The project was developed as part of the Foundations of Data Science course at ETH Zurich. 
The goal is to compare the performance of different models and identify the most effective approach for predicting CVD.

## Features
- Random Forest Classifier: A robust ensemble learning method for classification tasks.
- K-Nearest Neighbors (KNN): A simple, distance-based classification algorithm.
- Support Vector Machine (SVM): A powerful algorithm for linear and non-linear classification.
- Deep Neural Network (DNN): A multi-layer perceptron for learning complex patterns in the data.
- Model Comparison: Evaluate and compare the performance of all models using multiple model evaluation metrics. 

## Installation
1. Clone the repository:
```bash
git clone https://github.com/ValentinETHZ/FoDS_G05_CvD.git
cd FoDS_G05_CvD 
``` 
2. Install dependencies:
```bash
pip install -r requirements.txt  
```
3. Run the application:
```bash
python main.py
```
## Usage
- The dataset (`2025_cardio_train.csv`) is included in the `data/` folder. It contains patient information such as age, blood pressure, cholesterol levels, and more.
- The `main.py` script integrates all models and compares their performance.
- Each model is implemented in the `models/` folder as a separate Python file.
- The script outputs evaluation results and plots for each model.
- You can adjust hyperparameters for each model in their respective scripts (e.g., `RandomForest.py`, `KNN.py`).

### Evaluation Metrics
The models are evaluated using AUC, accuracy, precision, recall and F1 score, to provide a comprehensive assessment of predictive performance.

### Reproducibility
The script saves model results to `output/results.pkl` after the first run. On subsequent runs, results are loaded from this file for fast plotting and analysis. To rerun all models and feature importance analysis, delete `output/results.pkl` and run `python main.py` again.

### Data Exploration and Hyperparameter Tuning
Exploratory data analysis (EDA) and data overview are performed in support/data_overview_EDA.ipynb.
Hyperparameter tuning for each model is implemented in the support/ folder (e.g., rf_tuning.py, knn_tuning.py, etc.).
These scripts and notebooks are provided for transparency and reproducibility, but are not required to run the main pipeline.

## Data
The dataset contains 70,000 rows of patient data with the following features:

Age: Patient's age in days.
Gender: Male or female.
Height: Patient's height in cm.
Weight: Patient's weight in kg.
Blood Pressure: Systolic (ap_hi) and diastolic (ap_lo) blood pressure.
Cholesterol: Normal, above normal, or well above normal.
Glucose: Normal, above normal, or well above normal.
Lifestyle Factors: Smoking, alcohol consumption, and physical activity as binary variables. 
Target Variable: cardio (1 = CVD, 0 = No CVD).

### Outputs
- Model performance comparison bar chart (`output/model_performance_comparison.png`)
- ROC curve plot for all models (`output/roc_curves.png`)
- Feature importance plots for each model (`output/feature_importance_<model>.png`)
- Tabular summary of all model evaluation metrics (`output/model_metrics.csv`)


## Repository Structure: 
```bash
FoDS_G05_CvD/
│
├── data/                       # Raw data files
│   └── 2025_cardio_train.csv
│
├── models/                     # Final model functions with chosen parameters
│   ├── RandomForest.py
│   ├── KNN.py
│   ├── SVM.py
│   ├── DNN.py
│
├── output/                     # All generated outputs (plots, results, etc.)
│   ├── model_performance_comparison.png
│   ├── roc_curves.png
│   ├── feature_importance_dnn.png
│   ├── feature_importance_knn.png
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_svm.png
│   ├── aggregate_normalized_feature_importance.png
│   ├── results.pkl
│   ├── model_metrics.csv
│
├── support/                    # Support code for experiments and tuning
│   ├── data_overview.ipynb         # Data overview and EDA notebook
│   ├── rf_tuning.py                # Random Forest hyperparameter tuning
│   ├── knn_tuning.py               # KNN hyperparameter tuning
│   ├── svm_tuning.py               # SVM hyperparameter tuning
│   ├── dnn_tuning.py               # DNN hyperparameter tuning
│   ├── dnn_tuning.csv              # DNN tuning results 
│
├── main.py                     # Main script to run the full workflow
├── requirements.txt            # All dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Ignore unnecessary files (e.g., __pycache__, .DS_Store)
```  
