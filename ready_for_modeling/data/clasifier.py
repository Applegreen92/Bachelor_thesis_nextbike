import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import os

# Load the data from the CSV file
csv_file = 'combined_city_data.csv'  # Replace with the path to your CSV file
original_df = pd.read_csv(csv_file)


def convert_target(data, threshold):
    """Convert 'bikes_available' to a binary target variable based on a threshold."""
    return data['bikes_available'].apply(lambda x: 0 if x <= threshold else 1)


def train_and_evaluate_classifier(classifier_name, thresholds, selected_features, output_dir):
    """
    Train and evaluate the specified classifier with the given parameters.

    Args:
        classifier_name (str): The name of the classifier ('RandomForest', 'GradientBoosting', 'XGBoost').
        thresholds (list): List of thresholds to test.
        selected_features (list): List of feature names to include in the model.
        output_dir (str): Directory to save the output plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {'Threshold': [], 'Accuracy': [], 'Feature Importances': []}

    for threshold in thresholds:
        df = original_df.copy()
        df['target'] = convert_target(df, threshold)
        X = df[selected_features]
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if classifier_name == 'RandomForest':
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        elif classifier_name == 'GradientBoosting':
            classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        elif classifier_name == 'XGBoost':
            classifier = XGBClassifier(n_estimators=200, random_state=42, use_label_encoder=False,
                                       eval_metric='logloss')
        else:
            raise ValueError("Unsupported classifier. Choose from 'RandomForest', 'GradientBoosting', or 'XGBoost'.")

        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        feature_importances = classifier.feature_importances_

        results['Threshold'].append(threshold)
        results['Accuracy'].append(accuracy)
        results['Feature Importances'].append(feature_importances)

        print(f'Classifier: {classifier_name}, Threshold: {threshold}, Accuracy: {accuracy:.2f}')

        importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Feature Importances for {classifier_name} at Threshold {threshold}')
        plt.savefig(os.path.join(output_dir, f'{classifier_name}_feature_importance_threshold_{threshold}.png'))
        plt.close()

    results_df = pd.DataFrame(results)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Threshold', y='Accuracy', data=results_df)
    plt.title(f'Accuracy for Different Thresholds ({classifier_name})')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.savefig(os.path.join(output_dir, f'{classifier_name}_accuracy_for_different_thresholds.png'))
    plt.show()


def train_and_evaluate_regressor(selected_features, output_dir):
    """
    Train and evaluate a RandomForest Regressor with the given parameters.

    Args:
        selected_features (list): List of feature names to include in the model.
        output_dir (str): Directory to save the output plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = original_df.copy()
    X = df[selected_features]
    y = df['bikes_available']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    feature_importances = regressor.feature_importances_

    print(f'Regressor MSE: {mse:.2f}, R^2: {r2:.2f}')

    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importances for RandomForest Regressor')
    plt.savefig(os.path.join(output_dir, 'random_forest_regressor_feature_importance.png'))
    plt.close()

    validation_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    validation_df.to_csv(os.path.join(output_dir, 'random_forest_regressor_validation_data.csv'), index=False)


train_and_evaluate_classifier(
    classifier_name='RandomForest',
    thresholds=[0],
    selected_features=['lon', 'lat', 'bikes_returned', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'cloud_cover', 'precipitation'],
    output_dir='random_forest_output'
)

# train_and_evaluate_classifier(
#     classifier_name='GradientBoosting',
#     thresholds=[0],
#     selected_features=['lon', 'lat', 'bikes_returned', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'cloud_cover', 'sfcWind', 'precipitation'],
#     output_dir='gradient_boosting_output'
# )
#
# train_and_evaluate_classifier(
#     classifier_name='XGBoost',
#     thresholds=[0],
#     selected_features=['bikes_returned', 'bikes_difference', 'hour', 'day', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'cloud_cover', 'sfcWind', 'precipitation'],
#     output_dir='xgboost_output'
# )
#
# train_and_evaluate_regressor(
#     selected_features=['lon', 'lat', 'bikes_returned', 'bikes_difference', 'hour', 'month', 'weekday', 'is_weekend', 'is_holiday', 'temperature', 'cloud_cover', 'sfcWind', 'precipitation'],
#     output_dir='random_forest_regressor_output'
#)
