import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle


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

    # Extract feature names
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Scaling

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    

    return X_train, X_test, y_train, y_test, feature_names


def run_model(model_func, X_train, X_test, y_train, y_test, feature_names):
    # Call the model function
    y_pred, y_score, feature_importance = model_func(X_train, X_test, y_train, y_test, feature_names)

    # Evaluate the model
    auc = roc_auc_score(y_test, y_score)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        "AUC": auc,
        "Accuracy": accuracy,
        "Confusion Matrix": cm,
        "Precision": precision,
        "Recall": recall, "F1": f1,
        "Feature Importance": feature_importance,
        "y_score": y_score #for ROC curve plotting
    }

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    """
    # Run models and collect results
    results = {}
    results["Random Forest"] = run_model(RF_func, X_train, X_test, y_train, y_test, feature_names)
    results["KNN"] = run_model(KNN_func, X_train, X_test, y_train, y_test, feature_names)
    results["SVM"] = run_model(SVM_func, X_train, X_test, y_train, y_test, feature_names)
    results["DNN"] = run_model(DNN_func, X_train, X_test, y_train, y_test, feature_names)
    """
    # Run models and collect results
    results_path = "output/results.pkl"

    if os.path.exists(results_path):
        with open(results_path, "rb") as f:
            results = pickle.load(f)
        print("Loaded results from output/results.pkl")
    else:
        results = {}
        print("Running Random Forest...")
        results["Random Forest"] = run_model(RF_func, X_train, X_test, y_train, y_test, feature_names)
        print("Running KNN...")
        results["KNN"] = run_model(KNN_func, X_train, X_test, y_train, y_test, feature_names)
        print("Running SVM...")
        results["SVM"] = run_model(SVM_func, X_train, X_test, y_train, y_test, feature_names)
        print("Running DNN...")
        results["DNN"] = run_model(DNN_func, X_train, X_test, y_train, y_test, feature_names)
        with open(results_path, "wb") as f:
            pickle.dump(results, f)
        print("Saved results to output/results.pkl")

    # Print results
    print("\nModel Performance Comparison:")
    for model, metrics in results.items():
        print(f"{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}")

    print("Creating model performance bar chart...")
    #Visualize performance metrics
    metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1"]
    models = list(results.keys())

    #Gather metric values for each model
    metric_values = {metric: [results[model][metric] for model in models] for metric in metrics}

    x = np.arange(len(metrics))
    width = 0.18

    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        plt.bar(x + i*width, [metric_values[m][i] for m in metrics], width, label=model)
    plt.xticks(x + width*1.5, metrics)
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.savefig("output/model_performance_comparison.png", bbox_inches='tight')
    plt.show()
    print("Saved model performance bar chart.")

    print("Creating ROC curve plot...")
    #ROC Curve visualization
    plt.figure(figsize=(8, 6))
    for model in models:
        y_score = results[model]["y_score"]  
        # You need to store y_score in your results dict for each model in run_model
        fpr, tpr, _ = roc_curve(y_test, results[model]["y_score"])
        plt.plot(fpr, tpr, label=f"{model} (AUC = {results[model]['AUC']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig("output/roc_curves.png")
    plt.show()
    print("Saved ROC curve plot.")


    #Feature importance visualization
    print("Creating feature importance plots...")

    for model in models:
        feature_importance = results[model]["Feature Importance"]
        # feature_importance is a list of (feature_name, importance_value)
        # Sort by importance (descending)
        feature_importance_sorted = sorted(feature_importance, key=lambda x: x[1], reverse=True)
        feature_names, importances = zip(*feature_importance_sorted)
        plt.figure(figsize=(8, 5))
        plt.barh(range(len(importances)), importances[::-1], align='center')
        plt.yticks(range(len(importances)), feature_names[::-1])
        plt.xlabel('Importance')
        plt.title(f'Feature Importances: {model}')
        plt.tight_layout()
        plt.savefig(f"output/feature_importance_{model.replace(' ', '_').lower()}.png")
        plt.show()
    print("All feature importance plots saved.")