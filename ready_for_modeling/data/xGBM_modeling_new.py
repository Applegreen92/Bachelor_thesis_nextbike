import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['city_lat','city_lng','bike_racks','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind', 'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']
y_train_reg = train_df[target_col]  # Cap the target values at 40

# Initialize XGBoost Classifier with given hyperparameters
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

# Initialize XGBoost Regressor with given hyperparameters
# reg_xgb = XGBRegressor(
#     colsample_bytree=0.6977924525180372,
#     gamma=4.897754201908785,
#     learning_rate=0.03899301447596341,
#     max_depth=9,
#     min_child_weight=5,
#     n_estimators=265,
#     subsample=0.6657489216128507,
#     random_state=3
# )

# Fit the models on the training data
clf_xgb.fit(X_train, y_train_clf)
#reg_xgb.fit(X_train, y_train_reg)

# List of test CSV files
test_files = [
    'test_data/new_2022_combined_city_data.csv'
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
    y_test_reg = test_df[target_col]

    # Make predictions on the testing data
    clf_xgb_predictions = clf_xgb.predict(X_test)
    #reg_xgb_predictions = reg_xgb.predict(X_test)

    # Calculate the accuracy for the classifier
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Calculate the RMSE for the regressor
    #reg_xgb_rmse = np.sqrt(mean_squared_error(y_test_reg, reg_xgb_predictions))

    # Print results
    print(f'Results for {test_file}:')
    print(f'XGBoost Classifier Accuracy: {clf_xgb_accuracy:.4f}')
    #print(f'XGBoost Regressor RMSE: {reg_xgb_rmse:.4f}')
