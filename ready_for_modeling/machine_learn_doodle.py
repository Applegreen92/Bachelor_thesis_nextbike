import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

csv_files = [
    #'data/complete_dresden.csv',
    #'data/complete_essen.csv',
    #'data/complete_heidelberg.csv',
    #'data/complete_nürnberg.csv',
    'data/combined_city_data.csv'
]
output_folder = 'pictures/'

# Features to exclude from analysis
exclude_features = ['datetime', 'lat', 'lon','station_name', 'year']


# Function to plot feature importance
def plot_feature_importance(importance, names, model_type, city_name):
    print(f"Plotting feature importance for {city_name}...")
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(10, 8))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(f'Feature Importance in {city_name}')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.savefig(f'{output_folder}feature_importance_{city_name}.png')
    plt.close()
    print(f"Feature importance plot for {city_name} saved.")


# Function to plot heatmap of correlations
def plot_correlation_heatmap(df, city_name):
    print(f"Plotting heatmap of correlations for {city_name}...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f'Feature Correlation Heatmap in {city_name}')
    plt.savefig(f'{output_folder}heatmap_correlation_{city_name}.png')
    plt.close()
    print(f"Heatmap of correlations for {city_name} saved.")


# Analyze each CSV file
for csv_file in csv_files:

    # Load the data
    city_name = csv_file.split('_')[1].capitalize()
    df = pd.read_csv(csv_file)

    # Drop the excluded features
    df = df.drop(columns=exclude_features)

    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Define the target variable and features
    target = 'bikes_available'
    features = df.drop(columns=[target]).columns

    # Split the data

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a RandomForest model
    print(f"Training RandomForest model for {city_name}...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Get feature importance
    feature_importance = model.feature_importances_

    # Plot and save feature importance
    plot_feature_importance(feature_importance, features, 'Random Forest', city_name)

    # Plot and save heatmap of correlations
    plot_correlation_heatmap(df, city_name)

print("Analysis completed and images saved.")
