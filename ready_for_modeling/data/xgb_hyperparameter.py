import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import uniform, randint

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['bikes_booked','bikes_difference','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Initialize XGBoost Classifier
clf_xgb = XGBClassifier(
    random_state=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Define the parameter grid to search
param_dist = {
    'n_estimators': randint(50, 300),
    'learning_rate': uniform(0.01, 0.2),
    'max_depth': randint(3, 15),
    'subsample': uniform(0.5, 0.9),
    'colsample_bytree': uniform(0.5, 0.9),
    'gamma': uniform(0, 5),
    'min_child_weight': randint(1, 10),
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=clf_xgb,
    param_distributions=param_dist,
    n_iter=100,
    scoring='accuracy',
    n_jobs=-1,
    cv=5,
    verbose=1,
    random_state=3
)

# Fit the RandomizedSearchCV
random_search.fit(X_train, y_train_clf)

# Print the best parameters and the best score
print(f"Best Parameters: {random_search.best_params_}")
print(f"Best Cross-Validation Accuracy: {random_search.best_score_:.4f}")

# Use the best estimator to predict on the test set
best_clf_xgb = random_search.best_estimator_

# List of test CSV files
test_files = [
    '2022_combined_city_data.csv',
]

# Loop through each test file, evaluate the models, and print results
for test_file in test_files:
    print(f"\nEvaluating on {test_file}...")

    # Load the testing data
    test_df = pd.read_csv(test_file)

    # Transform the target variable for binary classification
    test_df['binary_bikes_available'] = (test_df[target_col] > 0).astype(int)

    # Separate features and binary target variable from testing data
    X_test = test_df[selected_features]
    y_test_clf = test_df['binary_bikes_available']

    # Make predictions on the testing data using the best model
    clf_xgb_predictions = best_clf_xgb.predict(X_test)

    # Calculate the accuracy for the classifier
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Print results
    print(f'XGBoost Classifier Accuracy with Tuned Hyperparameters: {clf_xgb_accuracy:.4f}')
