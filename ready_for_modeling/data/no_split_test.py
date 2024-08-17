import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

# Load the training data
train_df = pd.read_csv('test_data/new_combined_city_data.csv')

# Define feature columns and target column
selected_features = ['city_lat','city_lng','bike_racks','lon', 'lat', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'sfcWind',
                     'precipitation']
target_col = 'bikes_available'

# Transform the target variable for binary classification
train_df['binary_bikes_available'] = (train_df[target_col] > 0).astype(int)

# Separate features and binary target variable from training data
X_train = train_df[selected_features]
y_train_clf = train_df['binary_bikes_available']
y_train_reg = train_df[target_col]

# Initialize the RandomForestClassifier and RandomForestRegressor
clf_rf = RandomForestClassifier(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=42
)

reg_rf = RandomForestRegressor(
    n_estimators=50,
    min_samples_split=5,
    min_samples_leaf=4,
    max_features=None,
    max_depth=10,
    bootstrap=False,
    random_state=42
)

# Initialize Gradient Boosting Classifier with given hyperparameters
clf_gbm = GradientBoostingClassifier(
    subsample=0.6,
    n_estimators=200,
    max_depth=10,
    learning_rate=0.01,
    random_state=42
)

# Initialize XGBoost Classifier with given hyperparameters
clf_xgb = XGBClassifier(
    colsample_bytree=0.5765133157615585,
    gamma=2.8319073793101883,
    learning_rate=0.015406375121368878,
    max_depth=11,
    min_child_weight=1,
    n_estimators=70,
    subsample=0.9033844439161279,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Initialize XGBoost Regressor with given hyperparameters
reg_xgb = XGBRegressor(
    colsample_bytree=0.6977924525180372,
    gamma=4.897754201908785,
    learning_rate=0.03899301447596341,
    max_depth=9,
    min_child_weight=5,
    n_estimators=265,
    subsample=0.6657489216128507,
    random_state=42
)

# Fit the models on the training data
clf_rf.fit(X_train, y_train_clf)
reg_rf.fit(X_train, y_train_reg)  # Uncomment to fit the regressor

clf_gbm.fit(X_train, y_train_clf)
clf_xgb.fit(X_train, y_train_clf)
reg_xgb.fit(X_train, y_train_reg)  # Fit the XGBoost Regressor

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
    clf_rf_predictions = clf_rf.predict(X_test)
    reg_rf_predictions = reg_rf.predict(X_test)  # Uncomment to use the regressor

    clf_gbm_predictions = clf_gbm.predict(X_test)
    clf_xgb_predictions = clf_xgb.predict(X_test)
    reg_xgb_predictions = reg_xgb.predict(X_test)  # Use the XGBoost Regressor

    # Calculate the accuracy for the classifiers
    clf_rf_accuracy = accuracy_score(y_test_clf, clf_rf_predictions)
    clf_gbm_accuracy = accuracy_score(y_test_clf, clf_gbm_predictions)
    clf_xgb_accuracy = accuracy_score(y_test_clf, clf_xgb_predictions)

    # Calculate the mean squared error for the regressors
    reg_rf_mse = mean_squared_error(y_test_reg, reg_rf_predictions)
    reg_xgb_mse = mean_squared_error(y_test_reg, reg_xgb_predictions)

    # Print results
    print(f'Results for {test_file}:')
    print(f'Random Forest Classifier Accuracy: {clf_rf_accuracy:.4f}')
    print(f'Gradient Boosting Classifier Accuracy: {clf_gbm_accuracy:.4f}')
    print(f'XGBoost Classifier Accuracy: {clf_xgb_accuracy:.4f}')
    print(f'Regressor Mean Squared Error (RF): {reg_rf_mse:.4f}')
    print(f'Regressor Mean Squared Error (XGB): {reg_xgb_mse:.4f}')  # Display MSE for the XGBoost Regressor
