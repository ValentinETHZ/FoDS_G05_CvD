

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as sc 
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



#data from forked repo
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

#one hot
data_encoded= pd.get_dummies(data, drop_first=True)
X = data_encoded.drop(columns=["cardio_1"])
y = data_encoded["cardio_1"]

#split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#scaling
scaler = sc()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#SVM model
model = SVC(kernel="rbf", C=10, gamma=0.01)  
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:}")

cm1 = confusion_matrix(y_test, y_pred)

# Display it
disp = ConfusionMatrixDisplay(confusion_matrix=cm1)
disp.plot(cmap="Blues")
plt.show()

"""
#Hyperparamtetric tuning and grid search optimization
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.001, 0.01, 0.1],
    'kernel': ['linear', 'rbf']
}

grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2, n_jobs=-1)

grid.fit(X_train, y_train)

best_params = grid.best_params_
print(f"Best parameters: {best_params}")

y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with best parameters: {accuracy:}")



# model performance visualisation based on parameter tuning
top5 = y_pred.sort_values(by="mean_test_score", ascending=False).head(5)

plt.bar(range(5), top5["mean_test_score"])
plt.xticks(range(5), top5["params"], rotation=45, ha="right")
plt.ylabel("Mean Test Score")
plt.title("Top 5 Hyperparameter Combinations")
plt.tight_layout()
plt.show()
"""

