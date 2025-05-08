
from sklearn.svm import SVC

def svm(X_train, y_train, X_test):
    #from sklearn.svm import SVC
    model = SVC(kernel="rbf", C=10, gamma=0.01) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred