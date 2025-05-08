import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

### Data loading and preprocessing ###

# Data loading
data = pd.read_csv(
        filepath_or_buffer="../data/2025_cardio_train.csv",
        index_col=0,
        na_filter=False,
        dtype = {
            "age": "int",
            "height": "int",
            "weight": "float",
            "gender": "category",
            "ap_hi": "int",
            "ap_lo": "int",
            "cholesterol": "category",
            "gluc": "category",
            "smoke": "category",
            "alco": "category",
            "active": "category",
            "cardio": "category",
        }
)

# order ordinal columns
data["cholesterol"] = data["cholesterol"].cat.as_ordered()
data["gluc"] = data["gluc"].cat.as_ordered()

# Construct features and labels
X = data.drop('cardio', axis=1)
y = data['cardio']

# split into test and train set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2025)

# Apply data preprocessing; scaling numerical columns
sc = StandardScaler(with_mean=False, with_std=False)

num_cols = X.select_dtypes(include=['int', 'float']).columns

X_train[num_cols] = sc.fit_transform(X_train[num_cols])
X_test[num_cols] = sc.transform(X_test[num_cols])

# Convert pandas DataFrame to numpy array
X_train, X_test, y_train, y_test = (
    np.array(X_train),
    np.array(X_test),
    np.array(y_train),
    np.array(y_test),
)

### Linear Regression ###

# Initialize the model using sklearn
LinR = LinearRegression()
# Fit the linear regression model using sklearn
LinR.fit(X_train, y_train)

# getting model performance score
r2_score = r2_score(y_test, LinR.predict(X_test))
rmse = root_mean_squared_error(y_test, LinR.predict(X_test))

print(f'R2 score: {r2_score}, RMSE: {rmse}')

### Logistic Regression ###

# Initialize the model using sklearn
LogR = LogisticRegression()
# Fit the linear regression model using sklearn
LogR.fit(X_train, y_train)

# Predict class labels on the test set
y_pred = LogR.predict(X_test)

# Evaluate performance using classification metrics
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print('Confusion Matrix:')
print(cm)

### Hyperparameter tuning ###

# Define the model
model = Ridge()

# Define the hyperparameter search space
alphas = np.logspace(-4, 4, 20)
print(alphas)

# Define the cross-validation setting
kf = KFold(n_splits=5, shuffle=True, random_state=42)

best_alpha = None
best_rmse = float('inf')

best_model = Ridge(alpha=best_alpha)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE: {:.3f} ".format(rmse))