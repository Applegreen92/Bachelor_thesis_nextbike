import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

csv_files = [
    #'data/complete_dresden.csv',
    #'data/complete_essen.csv',
    #'data/complete_heidelberg.csv',
    #'data/complete_nürnberg.csv',
    'data/combined_city_data.csv'
]
output_folder = 'pictures/'

# Features to exclude from analysis
exclude_features = ['datetime', 'lat', 'lon', 'station_name', 'year']


# Function to plot feature importance
def plot_feature_importance(importance, names, model_type, city_name):
    print(f"Plotting feature importance for {city_name} ({model_type})...")
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    plt.title(f'Feature Importance in {city_name} ({model_type})')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Names')
    plt.savefig(f'{output_folder}feature_importance_{city_name}_{model_type}.png')
    plt.close()
    print(f"Feature importance plot for {city_name} ({model_type}) saved.")



def plot_correlation_heatmap(df, city_name):
    print(f"Plotting heatmap of correlations for {city_name}...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title(f'Feature Correlation Heatmap in {city_name}')
    plt.savefig(f'{output_folder}heatmap_correlation_{city_name}.png')
    plt.close()
    print(f"Heatmap of correlations for {city_name} saved.")



def plot_pair_plot(df, city_name):
    print(f"Plotting pair plot for {city_name}...")
    plt.figure(figsize=(12, 12))
    sns.pairplot(df)
    plt.title(f'Pair Plot in {city_name}')
    plt.savefig(f'{output_folder}pair_plot_{city_name}.png')
    plt.close()
    print(f"Pair plot for {city_name} saved.")



def plot_distribution_plot(df, city_name):
    print(f"Plotting distribution plot for {city_name}...")
    plt.figure(figsize=(12, 10))
    for column in df.columns:
        sns.histplot(df[column], kde=True)
        plt.title(f'Distribution Plot of {column} in {city_name}')
        plt.savefig(f'{output_folder}distribution_plot_{city_name}_{column}.png')
        plt.close()
    print(f"Distribution plots for {city_name} saved.")



def plot_box_plot(df, city_name):
    print(f"Plotting box plot for {city_name}...")
    plt.figure(figsize=(12, 10))
    sns.boxplot(data=df)
    plt.title(f'Box Plot in {city_name}')
    plt.xticks(rotation=90)
    plt.savefig(f'{output_folder}box_plot_{city_name}.png')
    plt.close()
    print(f"Box plot for {city_name} saved.")



def plot_scatter_plot(df, city_name, target):
    print(f"Plotting scatter plot for {city_name}...")
    plt.figure(figsize=(12, 10))
    for column in df.columns:
        if column != target:
            sns.scatterplot(x=df[column], y=df[target])
            plt.title(f'Scatter Plot of {column} vs {target} in {city_name}')
            plt.savefig(f'{output_folder}scatter_plot_{city_name}_{column}.png')
            plt.close()
    print(f"Scatter plots for {city_name} saved.")



for csv_file in csv_files:
    print(f"Processing {csv_file}...")
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

    # Train and analyze RandomForest model
    print(f"Training RandomForest model for {city_name}...")
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_feature_importance = rf_model.feature_importances_
    plot_feature_importance(rf_feature_importance, features, 'Random Forest', city_name)

    # Train and analyze GradientBoosting model
    print(f"Training GradientBoosting model for {city_name}...")
    gbm_model = GradientBoostingRegressor(random_state=42)
    gbm_model.fit(X_train, y_train)
    gbm_feature_importance = gbm_model.feature_importances_
    plot_feature_importance(gbm_feature_importance, features, 'GBM', city_name)

    # Train and analyze XGBoost model
    print(f"Training XGBoost model for {city_name}...")
    xgb_model = XGBRegressor(random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_feature_importance = xgb_model.feature_importances_
    plot_feature_importance(xgb_feature_importance, features, 'XGBoost', city_name)

    # Plot and save heatmap of correlations
    plot_correlation_heatmap(df, city_name)

    # Plot and save pair plot
    plot_pair_plot(df, city_name)

    # Plot and save distribution plot
    plot_distribution_plot(df, city_name)

    # Plot and save box plot
    plot_box_plot(df, city_name)

    # Plot and save scatter plot
    plot_scatter_plot(df, city_name, target)

print("Analysis completed and images saved.")
