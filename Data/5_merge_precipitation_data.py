import pandas as pd
import os
# Define file paths
provided_csv_path = 'weather/precipitation/preprocessed_precipitation_essen.csv'
target_csv_path = ('preprocessed_data/Checked_preprocessed_data/Essen'
                   '/windSpeed_cloudCover_temp_bike_availability_essen.csv')



# Load the provided CSV file
provided_df = pd.read_csv(provided_csv_path)

# Extract the necessary columns
precipitation_data = provided_df[['datetime', 'precipitation']]

# Load the target CSV file
target_df = pd.read_csv(target_csv_path)

# Merge the dataframes on the datetime column
merged_df = pd.merge(target_df, precipitation_data, on='datetime', how='left')

# Save the merged dataframe to a new CSV file

#creating an output name
base_name = os.path.basename(target_csv_path)
new_base_name = f"precipitation_{base_name}"
output_path = 'preprocessed_data/'
output_file_path = os.path.join(output_path, new_base_name)
merged_df.to_csv(output_file_path, index=False)

