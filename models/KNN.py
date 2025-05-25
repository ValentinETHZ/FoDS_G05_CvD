from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance


def KNN_func(X_train, X_test, y_train, y_test, feature_names):

    # Create a pipeline: scaling + KNN
    pipe = Pipeline([
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
    proba = pipe.predict_proba(X_test)
    y_score = proba[:, 1]
    y_pred = pipe.predict(X_test)
    print('KNN model predictions finished')

    # Feature analysis
    # Permutation importance on test set (drop in ROC AUC)
    result = permutation_importance(
        pipe, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc"
    )
    mean_imp = result.importances_mean  # array of shape (n_features,)

    # Zip, sort, and return
    feature_importance = list(zip(feature_names, mean_imp))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print('KNN model feature analysis finished')
    return y_pred, y_score, feature_importance
