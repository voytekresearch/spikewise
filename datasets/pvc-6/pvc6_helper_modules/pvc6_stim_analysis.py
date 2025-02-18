


from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold, cross_val_predict
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.utils import resample 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_1samp
import numpy as np



from pvc6_plotting import *

import warnings
warnings.filterwarnings('ignore')


#CATEGORICAL STIM PREDICTION


#Logistic regression to predict stimulation type
def logistic_regression_stim(X, y, X_train, X_test, y_train, y_test):
    
    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define logistic regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(multi_class='multinomial', solver='lbfgs'))
    ])
    
    # Train the model
    model.fit(X_train, y_train)
    
   
 

    return model

#SVM to predict stimulation type with grid search
def svm_stim(X, y, X_train, X_test, y_train, y_test):

    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Define pipeline with SVM classifier
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='linear', probability=True))  # Linear kernel for simplicity
    ])
    
    # Define hyperparameters for grid search
    param_grid = {
        'classifier__C': [0.1, 1, 10, 100]  # Regularization parameter
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    
    # Get the best model and its accuracy
    best_model = grid_search.best_estimator_
   


    return model

#Random forest to predict stimulation type with grid search. Option for getting average accuracy of multiple ranndom states 
def random_forest_stim(X, y, X_train, X_test, y_train, y_test, multiple_random_states=True):


    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    if multiple_random_states == False:

        # Define pipeline with Random Forest classifier
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=48))  # 100 decision trees
        ])
        
        # Define hyperparameters for grid search
        param_grid = {
            'classifier__max_depth': [None, 10, 20],  # Maximum depth of the tree
            'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'classifier__bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        }
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Get the best model and its accuracy
        best_model = grid_search.best_estimator_
        accuracy = best_model.score(X_test, y_test)
        return model
    else:
        # Define a range of random states
        random_states = [42, 48, 64, 128, 256]
        
        # List to store results
        accuracies = []
        
        for state in random_states:
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=state))
            ])

             # Define hyperparameters for grid search
            param_grid = {
            'classifier__max_depth': [None, 10, 20],  # Maximum depth of the tree
            'classifier__min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
            'classifier__min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
            'classifier__bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
            }
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            accuracies.append(best_model.score(X_test, y_test))

      

        return best_model, accuracies
                    

#PINK STIM PREDICTIONS

# Function to train the model with progress tracking
def train_model_with_progress(X_train, y_train, max_iter=100, regression_type = 'Ridge'):
    if regression_type == 'Ridge':
        model = Ridge(solver='saga', max_iter=max_iter, random_state=42)
        progress_bar = tqdm(range(max_iter), desc="Training Ridge Regression")
        for _ in progress_bar:
            model.max_iter += 1
            model.fit(X_train, y_train)
    elif regression_type == 'lasso':

        # Initialize Lasso Model
 
        model = Lasso(alpha=0.1, max_iter=max_iter, warm_start=True, random_state=42)  # warm_start=True allows continuation of training
        
        # Progress bar for training
        progress_bar = tqdm(range(max_iter), desc="Training Lasso Regression")
        for _ in progress_bar:
            model.max_iter += 1  # Incrementally increase the number of iterations
            model.fit(X_train, y_train)  # Fit model         

    return model


#Function for ridge regression


def run_ridge_regression_kfold(X, y, n_splits=5, random_state=42, bootstraps=1000):
    """
    Perform K-Fold Ridge Regression with bootstrapping and return model statistics.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target variable.
        n_splits (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        bootstraps (int): Number of bootstrap resamples.

    Returns:
        dict: Contains predictions, coefficients, R² scores, bootstrapped CIs, standard errors, and p-values.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    model = Ridge()
    y_pred_cv = cross_val_predict(model, X, y, cv=kf)  
    model.fit(X, y)  

    # Get Feature Coefficients
    coefficients = model.coef_
    feature_names = X.columns

    # Compute Cross-Validated R² Scores
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

    print(f"Cross-validated R-squared scores: {scores}")
    print(f"Average R-squared: {scores.mean():.3f} ± {scores.std():.3f}")

    # ------------------ BOOTSTRAPPING ------------------
    bootstrapped_coefs = []
    
    for _ in range(bootstraps):
        X_resampled, y_resampled = resample(X, y, random_state=None)  # Ensure different resampling
        model.fit(X_resampled, y_resampled)
        bootstrapped_coefs.append(model.coef_)
    
    bootstrapped_coefs = np.array(bootstrapped_coefs)

    # Compute statistics
    lower_bound = np.percentile(bootstrapped_coefs, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_coefs, 97.5, axis=0)
    standard_errors = np.std(bootstrapped_coefs, axis=0)

    # Compute p-values using a t-test
    p_values = np.array([ttest_1samp(bootstrapped_coefs[:, i], 0)[1] for i in range(bootstrapped_coefs.shape[1])])

    return {
        "y_pred_cv": y_pred_cv,
        "coefficients": coefficients,
        "feature_names": feature_names,
        "r2_scores": scores,
        "bootstrapped_coefs": bootstrapped_coefs,
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
        "standard_errors": standard_errors,
        "p_values": p_values
    }
