import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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







train_and_evaluate_classifier(
    classifier_name='RandomForest',
    thresholds=[1, 2, 3, 4, 5],
    selected_features=['bikes_booked','bikes_returned','minute','hour','day','month','year','weekday','is_weekend','is_holiday','temperature','cloud_cover','sfcWind','precipitation'],  # Replace with your feature names
    output_dir='all_random_forest_output'
)

train_and_evaluate_classifier(
    classifier_name='GradientBoosting',
    thresholds=[1, 2, 3, 4, 5],
    selected_features=['bikes_booked','bikes_returned','minute','hour','day','month','year','weekday','is_weekend','is_holiday','temperature','cloud_cover','sfcWind','precipitation'],  # Replace with your feature names
    output_dir='all_gradient_boosting_output'
)

train_and_evaluate_classifier(
    classifier_name='XGBoost',
    thresholds=[1, 2, 3, 4, 5],
    selected_features=['bikes_booked','bikes_returned','minute','hour','day','month','year','weekday','is_weekend','is_holiday','temperature','cloud_cover','sfcWind','precipitation'],  # Replace with your feature names
    output_dir='all_xgboost_output'
)
