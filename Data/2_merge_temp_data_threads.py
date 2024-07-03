import threading
import time

import netCDF4 as nc
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load the CSV file containing bike availability and station information
csv_file_path = 'preprocessed_data/Checked_preprocessed_data/Essen/bike_availability_essen.csv'
csv_df = pd.read_csv(csv_file_path)

# Convert 'datetime' column to datetime objects
csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])


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


def get_temperature_for_month(dataset, lon_array, lat_array, tas_array, valid_mask, lon, lat, datetime_obj):
    lat_idx, lon_idx = find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, valid_mask)
    print(f"Nearest indices - lat_idx: {lat_idx}, lon_idx: {lon_idx}")

    time_units = dataset.variables['time'].units
    time_calendar = dataset.variables['time'].calendar
    time_idx = nc.date2index(datetime_obj, dataset.variables['time'], select='nearest')
    print(f"Time index: {time_idx}")

    temperature = tas_array[time_idx, lat_idx, lon_idx]
    print(f"Extracted temperature: {temperature}")

    return temperature


def process_month(month, df):
    dataset = datasets[month]
    lon_array = dataset.variables['lon'][:]
    lat_array = dataset.variables['lat'][:]
    tas_array = dataset.variables['tas'][:]
    valid_mask = ~np.isnan(tas_array).all(axis=0)

    temperatures = []
    for index, row in df.iterrows():
        try:
            print(f"[{threading.current_thread().name}] Processing row {index}: {row['datetime']}, {row['lon']}, {row['lat']}")
            temperature = get_temperature_for_month(dataset, lon_array, lat_array, tas_array, valid_mask, row['lon'],
                                                    row['lat'], row['datetime'])
            temperatures.append(temperature)
        except IndexError as e:
            print(f"IndexError at row {index}: {e}")
            temperatures.append(np.nan)

    df['temperature'] = temperatures
    return df


# Define the path template and months for the NetCDF files
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
file_path_template = 'weather/temperature/temperature_{}.nc'

# Pre-load NetCDF datasets into a dictionary
datasets = {month: nc.Dataset(file_path_template.format(month)) for month in months}

# Split the dataframe by month
monthly_dfs = {month: csv_df[csv_df['datetime'].dt.strftime('%b').str.lower() == month] for month in months}

start_time = time.time()
# Process data in parallel using threads
with ThreadPoolExecutor(max_workers=12) as executor:
    future_to_month = {executor.submit(process_month, month, df): month for month, df in monthly_dfs.items()}
    results = []
    for future in as_completed(future_to_month):
        month = future_to_month[future]
        try:
            result = future.result()
            results.append(result)
        except Exception as exc:
            print(f"Exception occurred during processing month {month}: {exc}")

# Combine results back into a single DataFrame
final_df = pd.concat(results)

# Save the updated DataFrame with temperature data to a new CSV file
output_file_path = 'preprocessed_data/Checked_preprocessed_data/Germany/bike_availability_temperature_germany.csv'
final_df.to_csv(output_file_path, index=False)

# Close all opened NetCDF datasets
for dataset in datasets.values():
    dataset.close()

end_time = time.time()
execution_time = end_time - start_time
print(f"Total execution time: {execution_time:.2f} seconds")
# Display the first few rows of the updated DataFrame
print(final_df.head())