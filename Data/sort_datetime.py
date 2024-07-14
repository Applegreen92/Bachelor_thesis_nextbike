import pandas as pd

# Define the file paths
input_csv_path = 'weather/precipitation/preprocessed_precipitation_essen.csv'
output_sorted_csv_path = input_csv_path



# Load the CSV file
df = pd.read_csv(input_csv_path)

# Convert the datetime column to datetime type (if not already)
df['datetime'] = pd.to_datetime(df['datetime'])

# Sort the dataframe by the datetime column
sorted_df = df.sort_values(by='datetime')

# Save the sorted dataframe to a new CSV file
sorted_df.to_csv(output_sorted_csv_path, index=False)
