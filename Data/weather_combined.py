import os
import threading
import time
import netCDF4 as nc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed



# Load the CSV file containing bike availability and station information
csv_file_path = 'preprocessed_data/Checked_preprocessed_data/heidelberg/2022_heidelberg_station.csv'
print(f"Loading CSV data from {csv_file_path}")
csv_df = pd.read_csv(csv_file_path)
csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])

# Define functions to find nearest index and get data for a month
def find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, mask):
    lon_flat = lon_array.flatten()
    lat_flat = lat_array.flatten()
    mask_flat = mask.flatten()

    valid_indices = np.where(mask_flat)[0]
    lon_flat = lon_flat[valid_indices]
    lat_flat = lat_flat[valid_indices]

    dist = np.sqrt((lon_flat - lon) ** 2 + (lat_flat - lat) ** 2)
    nearest_idx = valid_indices[dist.argmin()]
    lat_idx, lon_idx = np.unravel_index(nearest_idx, lon_array.shape)
    return lat_idx, lon_idx

def get_data_for_month(dataset, lon_array, lat_array, data_array, valid_mask, lon, lat, datetime_obj):
    lat_idx, lon_idx = find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, valid_mask)
    time_idx = nc.date2index(datetime_obj, dataset.variables['time'], select='nearest')
    data_value = data_array[time_idx, lat_idx, lon_idx]
    return data_value

def process_month(month, df, data_type, variable_name):
    print(f"Processing {data_type} for month {month}")
    dataset = datasets[data_type][month]
    lon_array = dataset.variables['lon'][:]
    lat_array = dataset.variables['lat'][:]
    data_array = dataset.variables[variable_name][:]
    valid_mask = ~np.isnan(data_array).all(axis=0)

    data_values = []
    for index, row in df.iterrows():
        try:
            print(f"[{threading.current_thread().name}] Processing row {index} for {data_type}: {row['datetime']}, {row['lon']}, {row['lat']}")
            data_value = get_data_for_month(dataset, lon_array, lat_array, data_array, valid_mask, row['lon'], row['lat'], row['datetime'])
            data_values.append(data_value)
        except IndexError as e:
            print(f"IndexError at row {index}: {e}")
            data_values.append(np.nan)

    df[data_type] = data_values
    return df

# Define the path templates and variables for the NetCDF files
weather_data = {
    'temperature': ('weather_2022/temperature/temperature_{}.nc', 'tas'),
    'cloud_cover': ('weather_2022/cloud_cover/cloudcover_{}.nc', 'clt'),
    'wind_speed': ('weather_2022/wind_speed/wind_speed_{}.nc', 'sfcWind')
}

months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Preload NetCDF datasets into a dictionary
print("Preloading NetCDF datasets")
datasets = {
    data_type: {month: nc.Dataset(path_template.format(month)) for month in months}
    for data_type, (path_template, variable_name) in weather_data.items()
}

# Process each type of weather data in parallel
results = []
for data_type, (path_template, variable_name) in weather_data.items():
    print(f"Processing {data_type} data")
    monthly_dfs = {month: csv_df[csv_df['datetime'].dt.strftime('%b').str.lower() == month] for month in months}

    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_month = {
            executor.submit(process_month, month, df.copy(), data_type, variable_name): month
            for month, df in monthly_dfs.items()
        }
        for future in as_completed(future_to_month):
            month = future_to_month[future]
            try:
                result = future.result()
                results.append(result)
                print(f"Completed processing {data_type} for month {month}")
            except Exception as exc:
                print(f"Exception occurred during processing {data_type} for month {month}: {exc}")

# Combine results back into a single DataFrame
print("Combining results into a single DataFrame")
final_df = pd.concat(results, axis=1)
final_df = final_df.loc[:,~final_df.columns.duplicated()]
final_df = final_df.sort_values(by='datetime')

# Save the updated DataFrame with weather data to a new CSV file
output_path = 'preprocessed_data/'
output_file_path = os.path.join(output_path, 'final_weather_data_2022_heidelberg_station.csv')
print(f"Saving final DataFrame to {output_file_path}")
final_df.to_csv(output_file_path, index=False)

# Close all opened NetCDF datasets
print("Closing all NetCDF datasets")
for data_type_datasets in datasets.values():
    for dataset in data_type_datasets.values():
        dataset.close()




# Display the first few rows of the updated DataFrame
print("First few rows of the final DataFrame:")
print(final_df.head())
