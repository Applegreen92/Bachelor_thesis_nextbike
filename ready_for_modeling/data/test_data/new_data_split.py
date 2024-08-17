import pandas as pd

# Load the original CSV file
df = pd.read_csv('new_combined_city_data.csv')

# Ensure that the 'datetime' column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Group the data by 'city_lat' and 'city_lng'
grouped = df.groupby(['city_lat', 'city_lng'])

# Iterate over each group and save it to a new CSV file
for (lat, lng), group in grouped:
    # Sort the group by 'datetime'
    sorted_group = group.sort_values(by='datetime')

    # Create a filename based on the latitude and longitude
    filename = f'city_{lat}_{lng}.csv'

    # Save the sorted group to a CSV file
    sorted_group.to_csv(filename, index=False)

print("Sorted CSV files have been created for each city.")
