import time

import netCDF4 as nc
import numpy as np
import pandas as pd

# Load the CSV file containing bike availability and station information
csv_file_path = 'preprocessed_data/Checked_preprocessed_data/Berliner_Platz/bike_availability_essen_Berliner_Platz.csv'
csv_df = pd.read_csv(csv_file_path)

# Convert 'datetime' column to datetime objects
csv_df['datetime'] = pd.to_datetime(csv_df['datetime'])

def find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, mask):
    """
    Find the indices of the nearest valid longitude and latitude values in 2D arrays.

    Parameters:
    lon_array (numpy array): 2D array of longitudes.
    lat_array (numpy array): 2D array of latitudes.
    lon (float): The longitude value to find the nearest to.
    lat (float): The latitude value to find the nearest to.
    mask (numpy array): 2D mask array indicating valid data points.

    Returns:
    tuple: Indices of the nearest valid longitude and latitude in the 2D arrays.
    """
    # Flatten the arrays and the mask to simplify the search
    lon_flat = lon_array.flatten()
    lat_flat = lat_array.flatten()
    mask_flat = mask.flatten()

    # Filter out invalid points
    valid_indices = np.where(mask_flat)[0]
    lon_flat = lon_flat[valid_indices]
    lat_flat = lat_flat[valid_indices]

    # Calculate the distance for each valid point
    dist = np.sqrt((lon_flat - lon)**2 + (lat_flat - lat)**2)

    # Find the index of the minimum distance
    nearest_idx = valid_indices[dist.argmin()]

    # Convert the flat index back to 2D indices
    lat_idx, lon_idx = np.unravel_index(nearest_idx, lon_array.shape)

    return lat_idx, lon_idx

def get_temperature_for_month(dataset, lon_array, lat_array, tas_array, valid_mask, lon, lat, datetime_obj):
    """
    Retrieve temperature from a NetCDF dataset for the nearest valid coordinates and time.

    Parameters:
    dataset (netCDF4.Dataset): Opened NetCDF dataset.
    lon_array (numpy array): Array of longitudes.
    lat_array (numpy array): Array of latitudes.
    tas_array (numpy array): Array of temperatures.
    valid_mask (numpy array): Mask of valid data points.
    lon (float): Longitude of the desired location.
    lat (float): Latitude of the desired location.
    datetime_obj (datetime): Datetime object representing the desired time.

    Returns:
    float: Temperature at the nearest valid coordinates and time.
    """
    # Find the nearest valid indexes for the given longitude and latitude
    lat_idx, lon_idx = find_nearest_2d_with_mask(lon_array, lat_array, lon, lat, valid_mask)
    print(f"Nearest indices - lat_idx: {lat_idx}, lon_idx: {lon_idx}")

    # Handle time
    time_units = dataset.variables['time'].units
    time_calendar = dataset.variables['time'].calendar
    time_idx = nc.date2index(datetime_obj, dataset.variables['time'], select='nearest')
    print(f"Time index: {time_idx}")
    print(time_idx, lat_idx, lon_idx)
    # Extract the temperature at the nearest coordinates and time index
    temperature = tas_array[time_idx, lat_idx, lon_idx]
    print(f"Extracted temperature: {temperature}")

    return temperature
start_time = time.time()

# Define the path template and months for the NetCDF files
months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
file_path_template = 'weather/temperature/temperature_{}.nc'

# Pre-load NetCDF datasets into a dictionary
datasets = {month: nc.Dataset(file_path_template.format(month)) for month in months}

# Add temperature data to the CSV DataFrame
temperatures = []

# Load variables for the current month
current_month = None
lon_array, lat_array, tas_array, valid_mask = None, None, None, None

for index, row in csv_df.iterrows():
    month = row['datetime'].strftime('%b').lower()  # Extract the month

    if month != current_month:
        # Update the current month and load new variables
        current_month = month
        dataset = datasets[current_month]
        lon_array = dataset.variables['lon'][:]
        lat_array = dataset.variables['lat'][:]
        tas_array = dataset.variables['tas'][:]
        valid_mask = ~np.isnan(tas_array).all(axis=0)

    try:
        print(f"Processing row {index}: {row['datetime']}, {row['lon']}, {row['lat']}")
        temperature = get_temperature_for_month(dataset, lon_array, lat_array, tas_array, valid_mask, row['lon'], row['lat'], row['datetime'])  # Get the temperature
        temperatures.append(temperature)  # Store the temperature
    except IndexError as e:
        print(f"IndexError at row {index}: {e}")
        temperatures.append(np.nan)  # Append NaN if there is an error

# Add the temperature data as a new column to the CSV DataFrame
csv_df['temperature'] = temperatures

# Save the updated DataFrame with temperature data to a new CSV file
output_file_path = 'preprocessed_data/bike_availability_germany.csv'
csv_df.to_csv(output_file_path, index=False)

# Close all opened NetCDF datasets
for dataset in datasets.values():
    dataset.close()

end_time = time.time()
final_time = start_time - end_time
print(f'{final_time} seconds for completion')
# Display the first few rows of the updated DataFrame
print(csv_df.head())
