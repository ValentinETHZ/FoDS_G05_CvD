
from sklearn.svm import SVC

def SVM_func(X_train, X_test, y_train):
    #from sklearn.svm import SVC
    model = SVC(kernel="rbf", C=10, gamma=0.01, probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # calculating y_probabilities
    proba = model.predict_proba(X_test)
    y_score = proba[:, 1]

    print('SVM model finished')
    return y_pred, y_score