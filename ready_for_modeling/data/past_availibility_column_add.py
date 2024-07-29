import pandas as pd

# Load the CSV file
file_path = 'combined_city_data.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Ensure the DataFrame is sorted by timestamp components to maintain chronological order
# Assuming you don't have a timestamp column, sort by year, month, day, hour, and minute


# Create lagged features for bike availability from 30 minutes and 1 hour before
# Assuming the data frequency is in minutes, determine the number of rows to shift
# For example, if the data is recorded every minute, 30 rows back for 30 minutes, 60 rows back for 1 hour
# Adjust the shift value according to your data frequency
df['bikes_available_30m_ago'] = df['bikes_available'].shift(245)
df['bikes_available_1h_ago'] = df['bikes_available'].shift(490)

# Drop rows with NaN values generated by shifting
df = df.dropna()

# Save the updated DataFrame to a new CSV file
output_file_path = 'new_combined.csv'  # Replace with the desired output file path
df.to_csv(output_file_path, index=False)


print("Lagged features added and saved to:", output_file_path)
