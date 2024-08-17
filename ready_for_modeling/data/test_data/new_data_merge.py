import pandas as pd

# Step 1: Load the CSV files
combined_city_data = pd.read_csv('2022_combined_city_data.csv')
new_data = pd.read_csv('2022_new_data.csv')

# Step 2: Merge the DataFrames based on 'datetime' and 'station_name'
merged_data = pd.merge(combined_city_data, new_data[['datetime', 'station_name', 'city_lat', 'city_lng', 'bike_racks', 'free_racks']],
                       on=['datetime', 'station_name'],
                       how='left')

# Step 3: Sort the DataFrame (if needed)
# merged_data = merged_data.sort_values(by=['datetime', 'station_name']) # Uncomment if sorting is needed

# Step 4: Save the resulting DataFrame to a new CSV file
merged_data.to_csv('new_2022_combined_city_data.csv', index=False)

print("Data has been successfully merged and saved to 'combined_city_data.csv'.")
