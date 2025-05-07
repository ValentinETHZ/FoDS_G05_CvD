from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline



def knn_predicitons(X_train,X_test,y_train):

    # Create a pipeline: scaling + KNN
    pipe = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Standardize the features
    ('knn', KNeighborsClassifier())  # Step 2: Apply KNN
    ])

    # Set optimized parameters
    knn_metric = "manhattan"
    n_k_neighbours = 29
    knn_weights = "uniform"

    pipe.set_params(knn__n_neighbors=int(n_k_neighbours),
                knn__weights=knn_weights,
                knn__metric=knn_metric)

    # fit and return predictions
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return y_pred
