


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

# Bootstrapping function
def bootstrap_model(model, X_train, y_train, X_test, y_test, n_bootstraps=1000):
    bootstrapped_accuracies = []
    for _ in range(n_bootstraps):
        X_resampled, y_resampled = resample(X_train, y_train, random_state=None)
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        bootstrapped_accuracies.append(acc)
    return np.array(bootstrapped_accuracies)

# Logistic Regression with bootstrapping
def logistic_regression_stim(X, y, X_train, X_test, y_train, y_test):
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    accs = bootstrap_model(model, X_train, y_train, X_test, y_test)
    return model, accs

# SVM with Grid Search and bootstrapping
def svm_stim(X, y, X_train, X_test, y_train, y_test):
    param_grid = {'C': [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(kernel='linear', probability=True), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    accs = bootstrap_model(best_model, X_train, y_train, X_test, y_test)
    return best_model, accs

# Random Forest with Grid Search and bootstrapping
def random_forest_stim(X, y, X_train, X_test, y_train, y_test):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    accs = bootstrap_model(best_model, X_train, y_train, X_test, y_test)
    return best_model, accs


    return best_model, accuracies
                    

#PINK STIM PREDICTIONS (RIDGE REGRESSION TO PREDIC STIM FEATURES AND LOG ISI) 

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
        dict: Contains predictions, coefficients, R² scores, adjusted R² scores,
              bootstrapped CIs, standard errors, and p-values.
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

    # Compute Adjusted R-squared for K-Fold
    n_samples = X.shape[0]
    n_features = X.shape[1]
    adjusted_r2_scores = 1 - ((1 - scores) * (n_samples - 1) / (n_samples - n_features - 1))

    print(f"Cross-validated R-squared scores: {scores}")
    print(f"Average R-squared: {scores.mean():.3f} ± {scores.std():.3f}")
    print(f"Adjusted R-squared scores: {adjusted_r2_scores}")
    print(f"Average Adjusted R-squared: {adjusted_r2_scores.mean():.3f} ± {adjusted_r2_scores.std():.3f}")

    # ------------------ BOOTSTRAPPING ------------------
    bootstrapped_coefs = []
    bootstrapped_r2 = []
    bootstrapped_adjusted_r2 = []
    
    for _ in range(bootstraps):
        X_resampled, y_resampled = resample(X, y, random_state=None)  # Ensure different resampling
        model.fit(X_resampled, y_resampled)
        bootstrapped_coefs.append(model.coef_)
        
        # Calculate R² and Adjusted R² for bootstrap sample
        r2 = model.score(X_resampled, y_resampled)
        adjusted_r2 = 1 - ((1 - r2) * (n_samples - 1) / (n_samples - n_features - 1))
        
        bootstrapped_r2.append(r2)
        bootstrapped_adjusted_r2.append(adjusted_r2)
    
    bootstrapped_coefs = np.array(bootstrapped_coefs)
    bootstrapped_r2 = np.array(bootstrapped_r2)
    bootstrapped_adjusted_r2 = np.array(bootstrapped_adjusted_r2)

    # Compute statistics for coefficients
    lower_bound = np.percentile(bootstrapped_coefs, 2.5, axis=0)
    upper_bound = np.percentile(bootstrapped_coefs, 97.5, axis=0)
    standard_errors = np.std(bootstrapped_coefs, axis=0)

    # Compute p-values using a t-test
    p_values = np.array([ttest_1samp(bootstrapped_coefs[:, i], 0)[1] for i in range(bootstrapped_coefs.shape[1])])

    # Compute statistics for R² and Adjusted R²
    r2_mean = np.mean(bootstrapped_r2)
    r2_ci = np.percentile(bootstrapped_r2, [2.5, 97.5])
    adjusted_r2_mean = np.mean(bootstrapped_adjusted_r2)
    adjusted_r2_ci = np.percentile(bootstrapped_adjusted_r2, [2.5, 97.5])

    return {
        "y_pred_cv": y_pred_cv,
        "coefficients": coefficients,
        "feature_names": feature_names,
        "r2_scores": scores,
        "adjusted_r2_scores": adjusted_r2_scores,
        "bootstrapped_coefs": bootstrapped_coefs,
        "bootstrapped_r2": bootstrapped_r2,
        "bootstrapped_adjusted_r2": bootstrapped_adjusted_r2,
        "ci_lower": lower_bound,
        "ci_upper": upper_bound,
        "standard_errors": standard_errors,
        "p_values": p_values,
        "r2_mean": r2_mean,
        "r2_ci": r2_ci,
        "adjusted_r2_mean": adjusted_r2_mean,
        "adjusted_r2_ci": adjusted_r2_ci
    }
