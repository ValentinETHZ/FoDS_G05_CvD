## Cardiovascular Disease Prediction Using Machine Learning Models
This repository contains the implementation of multiple machine learning models to predict cardiovascular disease (CVD) based on patient data. 
The project was developed as part of the Foundations of Data Science course at ETH Zurich. 
The goal is to compare the performance of different models and identify the most effective approach for predicting CVD.

## Features
- Random Forest Classifier: A robust ensemble learning method for classification tasks.
- K-Nearest Neighbors (KNN): A simple, distance-based classification algorithm.
- Support Vector Machine (SVM): A powerful algorithm for linear and non-linear classification.
- Deep Neural Network (DNN): A multi-layer perceptron for learning complex patterns in the data.
- Model Comparison: Evaluate and compare the performance of all models using accuracy and confusion matrices.

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
ipython main_CoW_qt.py path/to/CTA/image.nii.gz path/to/MRA/image.nii.gz path/to/CTA/CoW/segmentation.nii.gz path/to/MRA/CoW/segmentation.nii.gz
```
## Usage
Data Input: The dataset (2025_cardio_train.csv) is included in the data folder. It contains patient information such as age, blood pressure, cholesterol levels, and more.

Run Models: The main.py script integrates all models and compares their performance.

Each model is implemented in the models/ folder as a separate Python file.
The script outputs accuracy scores and confusion matrices for each model.
Modify Parameters: You can adjust hyperparameters for each model in their respective scripts (e.g., random_forest.py, knn.py).

## Data
The dataset contains 70,000 rows of patient data with the following features:

Age: Patient's age in days.
Gender: Male or female.
Height: Patient's height in cm.
Weight: Patient's weight in kg.
Blood Pressure: Systolic (ap_hi) and diastolic (ap_lo) blood pressure.
Cholesterol: Normal, above normal, or well above normal.
Glucose: Normal, above normal, or well above normal.
Lifestyle Factors: Smoking, alcohol consumption, and physical activity.
Target Variable: cardio (1 = CVD, 0 = No CVD).


## Repository Structure: 
```bash
FoDS_G05_CvD/
│
├── data/                     # Folder for dataset
│   └── 2025_cardio_train.csv
|
├── models/                   # Folder for individual model scripts
│   ├── random_forest.py      # Random Forest model (your file)
│   ├── knn.py                # KNN model
│   ├── dnn.py                # Deep Neural Network model
│   ├── svm.py                # SVM model
│
├── main.py                   # Main script to integrate and compare models
├── requirements.txt          # List of dependencies (e.g., pandas, scikit-learn, etc.)
├── README.md                 # Documentation for the project
```  
