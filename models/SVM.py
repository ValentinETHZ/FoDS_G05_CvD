from sklearn.inspection import permutation_importance
from sklearn.svm import SVC


def SVM_func(X_train, X_test, y_train, y_test, feature_names):

    #from sklearn.svm import SVC
    model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)

    # fitting the model on training set
    model.fit(X_train, y_train)

    # get predictions
    y_pred = model.predict(X_test)
    # calculating y_probabilities
    proba = model.predict_proba(X_test)
    y_score = proba[:, 1]
    print('SVM model predictions finished')

    # Feature analysis
    # Permutation importance on test set (drop in ROC AUC)
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=5,
        random_state=42,
        scoring="roc_auc"
    )
    mean_imp = result.importances_mean  # array of shape (n_features,)

    # Zip, sort, and return
    feature_importance = list(zip(feature_names, mean_imp))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print('SVM model feature analysis finished')
    return y_pred, y_score, feature_importance