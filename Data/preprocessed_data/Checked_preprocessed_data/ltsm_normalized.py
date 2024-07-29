import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf
# Load the data from the CSV file
csv_file = 'combined_citys/combined_city_data.csv'  # Replace with the path to your CSV file
original_df = pd.read_csv(csv_file)

# Convert 'datetime' column to datetime object
original_df['datetime'] = pd.to_datetime(original_df['datetime'])


# Function to convert 'bikes_available' to a binary target variable based on a threshold
def convert_target(data, threshold):
    return data['bikes_available'].apply(lambda x: 0 if x <= threshold else 1)


# Function to encode time features as cyclical
def encode_time_features(df):
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['weekday_sin'] = np.sin(2 * np.pi * df['weekday'] / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df['weekday'] / 7)
    return df


# List of thresholds to test
thresholds = [1]

# Directory to save the plots
output_dir = 'feature_importance_plots_lstm'
os.makedirs(output_dir, exist_ok=True)

# Initialize a dictionary to store accuracies
results = {'Threshold': [], 'Accuracy': []}

# List of features to exclude
exclude_features = ['station_name', 'datetime', 'bikes_available','minute','hour','day','month','weekday','year', 'bikes_booked']

for threshold in thresholds:
    # Create a fresh copy of the original DataFrame for each threshold
    df = original_df.copy()

    # Convert 'bikes_available' to a binary target variable
    df['target'] = convert_target(df, threshold)

    # Encode time features as cyclical
    df = encode_time_features(df)

    # Drop the original 'bikes_available' column and exclude specified features
    X = df.drop(columns=exclude_features)
    y = df['target']

    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input data to be 3D [samples, time steps, features]
    time_steps = 10  # Number of time steps
    X_lstm = []
    y_lstm = []
    for i in range(time_steps, len(X_scaled)):
        X_lstm.append(X_scaled[i - time_steps:i, :])
        y_lstm.append(y.iloc[i])

    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Predict the target values for the test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Store the results
    results['Threshold'].append(threshold)
    results['Accuracy'].append(accuracy)

    # Print the accuracy
    print(f'Threshold: {threshold}, Accuracy: {accuracy:.2f}')

# Convert results to DataFrame for easier plotting
results_df = pd.DataFrame(results)

# Plot the accuracy for different thresholds
plt.figure(figsize=(10, 6))
sns.barplot(x='Threshold', y='Accuracy', data=results_df)
plt.title('Accuracy for Different Thresholds')
plt.xlabel('Threshold')
plt.ylabel('Accuracy')
plt.savefig(os.path.join(output_dir, 'accuracy_for_different_thresholds.png'))
plt.show()
