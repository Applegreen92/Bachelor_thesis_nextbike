import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the training data
train_df = pd.read_csv('combined_city_data.csv')

# Define feature columns and target column
selected_features = ['lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']

# Initialize XGBoost Classifier with the first set of hyperparameters


# Initialize XGBoost Classifier with the second set of hyperparameters
clf_xgb = XGBClassifier(
    colsample_bytree=0.5765133157615585,
    gamma=2.8319073793101883,
    learning_rate=0.015406375121368878,
    max_depth=11,
    min_child_weight=1,
    n_estimators=70,
    subsample=0.9033844439161279,
    random_state=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Fit the models on the training data

clf_xgb.fit(X_train, y_train_clf)

# List of test CSV files
test_files = [
    '2022_combined_city_data.csv',
    '2022_complete_dresden.csv',
    '2022_complete_essen.csv',
    '2022_complete_heidelberg.csv',
    '2022_complete_nürnberg.csv'
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

    # Make predictions on the testing data with both models
    clf_xgb_predictions = clf_xgb.predict(X_test)

    # Calculate the accuracy for both classifiers

    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Print results
    print(f'Results for {test_file}:')

    print(f'XGBoost Classifier 2 Accuracy: {clf_xgb_accuracy:.4f}')
