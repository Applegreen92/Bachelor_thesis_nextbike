import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error

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
y_train_reg = train_df[target_col].clip(upper=40)  # Cap the target values at 40

# Initialize the RandomForestClassifier and RandomForestRegressor
clf_rf = RandomForestClassifier(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=3
)

reg_rf = RandomForestRegressor(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=3
)


# Initialize Gradient Boosting Classifier with given hyperparameters
clf_gbm = GradientBoostingClassifier(
    subsample=0.6,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.01,
    random_state=3
)

# Initialize XGBoost Classifier with given hyperparameters
clf_xgb = XGBClassifier(
    subsample=0.6,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.01,
    random_state=3,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Fit the models on the training data
clf_rf.fit(X_train, y_train_clf)
reg_rf.fit(X_train, y_train_reg)  # Uncomment to fit the regressor

clf_gbm.fit(X_train, y_train_clf)
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
    y_test_reg = test_df[target_col].clip(upper=40)  # Cap the target values at 40

    # Make predictions on the testing data
    clf_rf_predictions = clf_rf.predict(X_test)
    reg_rf_predictions = reg_rf.predict(X_test)  # Uncomment to use the regressor

    clf_gbm_predictions = clf_gbm.predict(X_test)
    clf_xgb_predictions = clf_xgb.predict(X_test)

    # Calculate the accuracy for the classifiers
    clf_rf_accuracy = accuracy_score(y_test_clf, clf_rf_predictions)
    clf_gbm_accuracy = accuracy_score(y_test_clf, clf_gbm_predictions)
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Calculate the mean squared error for the regressor (if needed)
    reg_rf_mse = mean_squared_error(y_test_reg, reg_rf_predictions)

    # Print results
    print(f'Results for {test_file}:')
    print(f'Random Forest Classifier Accuracy: {clf_rf_accuracy:.4f}')
    print(f'Gradient Boosting Classifier Accuracy: {clf_gbm_accuracy:.4f}')
    print(f'XGBoost Classifier Accuracy: {clf_xgb_accuracy:.4f}')
    print(f'Regressor Mean Squared Error: {reg_rf_mse:.4f}')  # Uncomment to display MSE for the regressor
