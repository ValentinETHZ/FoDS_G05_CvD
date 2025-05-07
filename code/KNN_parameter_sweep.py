import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sts
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

data = pd.read_csv(
        filepath_or_buffer="../data/2025_cardio_train.csv",
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

data_clone = data.copy()
data = data[data["ap_hi"] <= 200 ]
data = data[data["ap_hi"] >= 0]
data = data[data["ap_lo"] <= 200]
data = data[data["ap_lo"] >= 0]


data_encoded = pd.get_dummies(data, drop_first=True, )

X = data_encoded.drop("cardio_1", axis=1)
y = data_encoded["cardio_1"]

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a pipeline: scaling + KNN
pipe = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('knn', KNeighborsClassifier())  # Step 2: Apply KNN
])

# Define parameter grid for hyperparameter tuning
param_grid = [
    # Grid for 'euclidean' and 'manhattan' — no need for 'p'
    {
        'knn__n_neighbors': list(range(1, 31)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['euclidean', 'manhattan']
    },
    # Grid for 'minkowski' with p values
    {
        'knn__n_neighbors': list(range(1, 31)),
        'knn__weights': ['uniform', 'distance'],
        'knn__metric': ['minkowski'],
        'knn__p': [1, 2]
    }
]

# Create GridSearchCV with the pipeline
grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the model
grid_search.fit(X_train, y_train)

# Output best parameters and score
print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Evaluate on the test set
y_pred = grid_search.best_estimator_.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")



"""
# Create and train KNN model
param_grid = [
    # Grid for 'euclidean' and 'manhattan' — no need for 'p'
    {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    },
    # Grid for 'minkowski' with p values
    {
        'n_neighbors': list(range(1, 31)),
        'weights': ['uniform', 'distance'],
        'metric': ['minkowski'],
        'p': [1, 2]
    }
]

knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print("Best params:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# my program


neighbours = np.linspace(2,40,40)
accuracy_max = 0

for i in range(len(neighbours)):
    knn = KNeighborsClassifier(n_neighbors=int(neighbours[i]))
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > accuracy_max:
        accuracy_max = accuracy
        neighbour_iloc = i
print(f"Accuracy for {neighbour_iloc}: {accuracy_max:.4f}")

"""



