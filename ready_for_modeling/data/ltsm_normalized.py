import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import EarlyStopping
import tensorflow as tf

# Load the data from the CSV files
train_csv_file = 'combined_city_data.csv'  # Replace with the path to your training CSV file
test_csv_file = '2022_combined_city_data.csv'  # Replace with the path to your testing CSV file

train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

# Convert 'datetime' column to datetime object
train_df['datetime'] = pd.to_datetime(train_df['datetime'])
test_df['datetime'] = pd.to_datetime(test_df['datetime'])

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
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()

    # Convert 'bikes_available' to a binary target variable
    train_df_copy['target'] = convert_target(train_df_copy, threshold)
    test_df_copy['target'] = convert_target(test_df_copy, threshold)

    # Encode time features as cyclical
    train_df_copy = encode_time_features(train_df_copy)
    test_df_copy = encode_time_features(test_df_copy)

    # Drop the original 'bikes_available' column and exclude specified features
    X_train = train_df_copy.drop(columns=exclude_features)
    y_train = train_df_copy['target']
    X_test = test_df_copy.drop(columns=exclude_features)
    y_test = test_df_copy['target']

    # Normalize the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape input data to be 3D [samples, time steps, features]
    time_steps = 10  # Number of time steps
    X_train_lstm = []
    y_train_lstm = []
    for i in range(time_steps, len(X_train_scaled)):
        X_train_lstm.append(X_train_scaled[i - time_steps:i, :])
        y_train_lstm.append(y_train.iloc[i])

    X_train_lstm, y_train_lstm = np.array(X_train_lstm), np.array(y_train_lstm)

    X_test_lstm = []
    y_test_lstm = []
    for i in range(time_steps, len(X_test_scaled)):
        X_test_lstm.append(X_test_scaled[i - time_steps:i, :])
        y_test_lstm.append(y_test.iloc[i])

    X_test_lstm, y_test_lstm = np.array(X_test_lstm), np.array(y_test_lstm)

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(LSTM(units=50))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

    # Predict the target values for the test set
    y_pred_prob = model.predict(X_test_lstm)
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Calculate the accuracy
    accuracy = accuracy_score(y_test_lstm, y_pred)

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
