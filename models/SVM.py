
from sklearn.svm import SVC

def SVM_func(X_train, X_test, y_train):
    #from sklearn.svm import SVC
    model = SVC(kernel="rbf", C=10, gamma=0.01) 
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred