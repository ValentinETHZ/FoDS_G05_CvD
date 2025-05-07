
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline




def RF_func(X_train,X_test,y_train):

    #Standardization and scaling
    # Create a pipeline with StandardScaler and RandomForestClassifier
    pipe = Pipeline([
        ('scaler', StandardScaler()),  # Step 1: Standardize the features 
        ('rf', RandomForestClassifier (random_state = 42))  # Step 2: Apply Random Forest
    ])

    #set found hyperparameters
    n_estimators = 200
    max_depth = 20
    min_samples_split = 10
    min_samples_leaf = 4
    criterion = 'entropy'


    pipe.set_params(   
        rf__n_estimators=n_estimators, # Number of trees in the forest
        rf__max_depth=max_depth, # Maximum depth of the tree
        rf__min_samples_split=min_samples_split,  # Minimum number of samples required to split an internal node
        rf__min_samples_leaf=min_samples_leaf, # Minimum number of samples required to be at a leaf node
        rf__criterion=criterion # Function to measure the quality of a split
    )

    # fit and return predicitons
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    return y_pred






