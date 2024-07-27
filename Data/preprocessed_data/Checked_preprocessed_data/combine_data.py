import pandas as pd

# Define the file paths
file_paths = [
    'dresden/complete_dresden.csv',
    'essen/complete_essen.csv',
    'heidelberg/complete_heidelberg.csv',
    'nürnberg/complete_nürnberg.csv'
]

# Initialize a list to store the dataframes and a variable to count the total rows
dataframes = []
total_rows = 0

# Read each CSV file and count the number of rows
for file_path in file_paths:
    df = pd.read_csv(file_path)
    row_count = len(df)
    print(f"{file_path} has {row_count} rows.")
    total_rows += row_count
    dataframes.append(df)

# Combine the dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

combined_df = combined_df.sort_values(by='datetime')

# Save the combined dataframe to a new CSV file
output_path = 'combined_citys/combined_city_data.csv'
combined_df.to_csv(output_path, index=False)

# Verify the number of rows in the output file
combined_row_count = len(combined_df)
print(f"Combined dataframe has {combined_row_count} rows.")

# Check if the total number of rows matches the sum of the rows in the individual files
if combined_row_count == total_rows:
    print("The output file has the correct number of rows.")
else:
    print("There is a mismatch in the number of rows.")
