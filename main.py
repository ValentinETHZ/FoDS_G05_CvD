import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.RandomForest import RF_func
from models.KNN import KNN_func
from models.DNN import DNN_func
from models.SVM import SVM_func

def load_data():
    # Load the dataset
    data = pd.read_csv(
        filepath_or_buffer="data/2025_cardio_train.csv",
        index_col=0,
        na_filter=False,
        dtype = {"gender": "category",
                 "cholesterol": "category",
                 "gluc": "category",
                 "smoke": "category",
                 "alco": "category",
                 "active": "category",
                 "cardio": "category",
        }
    )
    
    data["cholesterol"] = data["cholesterol"].cat.as_ordered()
    data["gluc"] = data["gluc"].cat.as_ordered()

    # Outlier removal
    data = data[(data["ap_hi"] <= 200) & (data["ap_hi"] >= 0)]
    data = data[(data["ap_lo"] <= 200) & (data["ap_lo"] >= 0)]

    # One-hot encoding
    data_encoded = pd.get_dummies(data, drop_first=True)

    # Train-test split
    X = data_encoded.drop("cardio_1", axis=1)
    y = data_encoded["cardio_1"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def run_model(model_func, X_train, X_test, y_train, y_test):
    # Call the model function
    y_pred = model_func(X_train, X_test, y_train)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return {"Accuracy": accuracy, "Confusion Matrix": cm}

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Run models and collect results
    results = {}
    results["Random Forest"] = run_model(RF_func, X_train, X_test, y_train, y_test)
    results["KNN"] = run_model(KNN_func, X_train, X_test, y_train, y_test)
    results["DNN"] = run_model(DNN_func, X_train, X_test, y_train, y_test)
    results["SVM"] = run_model(SVM_func, X_train, X_test, y_train, y_test)

    # Print results
    print("\nModel Performance Comparison:")
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")