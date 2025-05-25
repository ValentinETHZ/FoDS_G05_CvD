from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline




def RF_func(X_train, X_test, y_train, y_test, feature_names):

    #Standardization and scaling
    # Create a pipeline with StandardScaler and RandomForestClassifier
    pipe = Pipeline([
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
    proba = pipe.predict_proba(X_test)
    y_score = proba[:, 1]
    y_pred = pipe.predict(X_test)
    print('RF model predictions finished')

    # Feature Analysis
    rf_model = pipe.named_steps['rf']
    importances = rf_model.feature_importances_

    # Zip with feature names & sort descending
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    print('RF model feature analysis finished')
    return y_pred, y_score, feature_importance






