import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


def plot_average_bikes_per_station(data_file, output_dir):
    """
    Plots the average bike availability for each station per hour over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by station name and time, then calculate the mean bike availability
    hourly_availability = df.groupby(['station_name', 'time'])['bikes_available'].mean().reset_index()

    # Pivot the data to have times as rows and stations as columns for plotting
    pivot_df = hourly_availability.pivot(index='time', columns='station_name', values='bikes_available')

    # Plot the average bikes available per hour for each station
    plt.figure(figsize=(18, 6))

    for station in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[station], marker='o', linestyle='-', label=station)

    plt.title('Average Bikes Available per Hour for Each Station')
    plt.xlabel('Time of the Day')
    plt.ylabel('Average Bikes Available')
    plt.xticks(rotation=45)
    plt.legend(title='Station')
    plt.tight_layout()
    plt.savefig(f'{output_dir}average_bikes_available_per_hour_each_station.png')
    plt.show()


def plot_average_bikes_all_stations_combined(data_file, output_dir):
    """
    Plots the average bike availability for all stations combined per hour over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by time to calculate the mean bike availability across all stations
    hourly_availability = df.groupby('time')['bikes_available'].mean().reset_index()

    # Plot the average bikes available per hour for all stations combined
    plt.figure(figsize=(18, 6))
    plt.plot(hourly_availability['time'], hourly_availability['bikes_available'], marker='o', linestyle='-')
    plt.title('Average Bikes Available per Hour for All Stations')
    plt.xlabel('Time of the Day')
    plt.ylabel('Average Bikes Available')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}average_bikes_available_per_hour_all_stations_combined.png')
    plt.show()

def plot_normalized_bikes_and_bookings(data_file, output_dir):
    """
    Plots the normalized average bike availability and average bikes booked for all stations combined per hour over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by time to calculate the mean bike availability and bikes booked across all stations
    hourly_availability = df.groupby('time')['bikes_available'].mean().reset_index()
    hourly_bookings = df.groupby('time')['bikes_booked'].mean().reset_index()

    # Merge the two dataframes on 'time'
    merged_df = pd.merge(hourly_availability, hourly_bookings, on='time')

    # Normalize the data
    scaler = MinMaxScaler()
    merged_df[['bikes_available', 'bikes_booked']] = scaler.fit_transform(merged_df[['bikes_available', 'bikes_booked']])

    # Plot the normalized average bikes available and bikes booked per hour for all stations combined
    plt.figure(figsize=(18, 6))
    plt.plot(merged_df['time'], merged_df['bikes_available'], marker='o', linestyle='-', label='Bikes Available')
    plt.plot(merged_df['time'], merged_df['bikes_booked'], marker='o', linestyle='-', label='Bikes Booked')
    plt.title('Normalized Average Bikes Available and Bikes Booked per Hour for All Stations')
    plt.xlabel('Time of the Day')
    plt.ylabel('Normalized Average Number')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}normalized_bikes_available_and_booked_per_hour_all_stations.png')
    plt.show()

def plot_normalized_bikes_availability_bookings_and_returns(data_file, output_dir):
    """
    Plots the normalized average bike availability, bikes booked, and bikes returned for all stations combined per hour over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by time to calculate the mean bike availability, bikes booked, and bikes returned across all stations
    hourly_data = df.groupby('time')[['bikes_available', 'bikes_booked', 'bikes_returned']].mean().reset_index()

    # Normalize the data
    scaler = MinMaxScaler()
    hourly_data[['bikes_available', 'bikes_booked', 'bikes_returned']] = scaler.fit_transform(hourly_data[['bikes_available', 'bikes_booked', 'bikes_returned']])

    # Plot the normalized average bikes available, bikes booked, and bikes returned per hour for all stations combined
    plt.figure(figsize=(18, 6))
    plt.plot(hourly_data['time'], hourly_data['bikes_available'], marker='o', linestyle='-', label='Bikes Available')
    plt.plot(hourly_data['time'], hourly_data['bikes_booked'], marker='o', linestyle='-', label='Bikes Booked')
    plt.plot(hourly_data['time'], hourly_data['bikes_returned'], marker='o', linestyle='-', label='Bikes Returned')
    plt.title('Normalized Average Bikes Available, Bikes Booked, and Bikes Returned per Hour for All Stations')
    plt.xlabel('Time of the Day')
    plt.ylabel('Normalized Average Number')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}normalized_bikes_availability_bookings_and_returns_per_hour_all_stations.png')
    plt.show()


def plot_average_bikes_and_counts(data_file, count_file, output_dir):
    """
    Plots the average bike availability for all stations combined per hour over the year
    and overlays the count of rows for each half-hour timeframe from a separate CSV file.

    Parameters:
    data_file (str): Path to the CSV file containing the bike availability data.
    count_file (str): Path to the CSV file containing the count of rows for each half-hour timeframe.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the bike availability data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by time to calculate the mean bike availability across all stations
    hourly_availability = df.groupby('time')['bikes_available'].mean().reset_index()

    # Load the count data
    count_df = pd.read_csv(count_file)

    # Convert the time column in the count data to string format
    count_df['datetime'] = pd.to_datetime(count_df['datetime'])
    count_df['time'] = count_df['datetime'].dt.strftime('%H:%M')

    # Calculate the count of rows for each half-hour timeframe
    time_counts = count_df.groupby('time').size().reset_index(name='count')

    # Normalize the data
    scaler = MinMaxScaler()
    hourly_availability['bikes_available'] = scaler.fit_transform(hourly_availability[['bikes_available']])
    time_counts['count'] = scaler.fit_transform(time_counts[['count']])

    # Plot the average bikes available per hour for all stations combined
    plt.figure(figsize=(18, 6))
    plt.plot(hourly_availability['time'], hourly_availability['bikes_available'], marker='o', linestyle='-',
             label='Average Bikes Available')

    # Overlay the count of rows for each half-hour timeframe
    plt.plot(time_counts['time'], time_counts['count'], marker='o', linestyle='-', label='Bikes without Station')

    plt.title('Average Bikes available with and without station')
    plt.xlabel('Time of the Day')
    plt.ylabel('Normalized Average Bikes Available')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}average_bikes_available_and_count_per_hour_all_stations_Nürnberg.png')
    plt.show()



def plot_annual_bikes_and_counts(data_file, count_file, output_dir):
    """
    Plots the average bike availability for all stations combined per day over the year
    and overlays the count of rows for each day from a separate CSV file.

    Parameters:
    data_file (str): Path to the CSV file containing the bike availability data.
    count_file (str): Path to the CSV file containing the count of rows for each day.
    output_dir (str): Directory where the plots will be saved.
    """
    # Load the bike availability data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the date component
    df['date'] = df['datetime'].dt.date

    # Group by date to calculate the mean bike availability across all stations
    daily_availability = df.groupby('date')['bikes_available'].mean().reset_index()

    # Load the count data
    count_df = pd.read_csv(count_file)

    # Convert the time column in the count data to datetime format
    count_df['datetime'] = pd.to_datetime(count_df['datetime'])
    count_df['date'] = count_df['datetime'].dt.date

    # Calculate the count of rows for each day
    daily_counts = count_df.groupby('date').size().reset_index(name='count')

    # Normalize the data
    scaler_avail = MinMaxScaler()
    scaler_count = MinMaxScaler()
    daily_availability['bikes_available'] = scaler_avail.fit_transform(daily_availability[['bikes_available']])
    daily_counts['count'] = scaler_count.fit_transform(daily_counts[['count']])

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots(figsize=(18, 6))

    # Plot the average bikes available per day for all stations combined on the primary y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Normalized Average Bikes Available', color=color)
    ax1.plot(daily_availability['date'], daily_availability['bikes_available'], marker='o', linestyle='-', color=color, label='Average Bikes Available')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a secondary y-axis to plot the count of rows for each day
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Normalized Count of Bikes without Station', color=color)
    ax2.plot(daily_counts['date'], daily_counts['count'], marker='o', linestyle='-', color=color, label='Bikes without Station')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add title and legend
    fig.suptitle('Average Bikes Available and Counted Bikes per Day Over the Year')
    fig.tight_layout()
    fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

    # Save the plot
    plt.savefig(f'{output_dir}average_bikes_available_and_count_per_day_all_stations_overlay.png')
    plt.show()


#plot_annual_bikes_and_counts('path/to/data_file.csv', 'path/to/count_file.csv', 'path/to/output_dir/')



def plot_normalized_daily_means_weather_data_with_bike_availibility(file_path):
    """
    Reads the dataset, computes daily mean values, normalizes them, and plots
    temperature, cloud cover, wind speed, and bike availability.

    Parameters:
    file_path (str): The path to the CSV file containing the dataset.
    """
    # Step 1: Read the dataset
    data = pd.read_csv(file_path)

    # Step 2: Preprocess the data to compute daily mean values
    data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour', 'minute']])
    data.set_index('datetime', inplace=True)

    # Compute daily mean values
    #daily_data = data.resample('D').mean()

    # Select the relevant columns
    features = ['temperature', 'bikes_booked']
    daily_mean = data[features].resample('D').mean()

    # Step 3: Normalize the data
    scaler = MinMaxScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(daily_mean), columns=features, index=daily_mean.index)

    # Step 4: Plot the data
    plt.figure(figsize=(36, 7))
    for column in normalized_data.columns:
        plt.plot(normalized_data.index, normalized_data[column], label=column)

    plt.title('Daily Mean Values of Temperature, Cloud Cover, Wind Speed, and Bike Availability (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.show()



import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_smoothed_monthly_mean_temp_vs_bikes_booked(data_file, output_dir, window_size=3):
    """
    Plots the smoothed monthly mean temperature and the mean bikes booked over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    window_size (int): The window size for the rolling mean to smooth the curves. Default is 3.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Extract the month from the datetime column
    df['month'] = df['datetime'].dt.month

    # Group by month to calculate the mean temperature and mean bikes booked
    monthly_data = df.groupby('month')[['temperature', 'bikes_booked']].mean().reset_index()

    # Apply a rolling mean to smooth the data
    monthly_data['temperature'] = monthly_data['temperature'].rolling(window=window_size, center=True).mean()
    monthly_data['bikes_booked'] = monthly_data['bikes_booked'].rolling(window=window_size, center=True).mean()

    # Normalize the data
    scaler = MinMaxScaler()
    monthly_data[['temperature', 'bikes_booked']] = scaler.fit_transform(monthly_data[['temperature', 'bikes_booked']])

    # Plot the smoothed monthly mean temperature and mean bikes booked
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_data['month'], monthly_data['temperature'], marker='o', linestyle='-', label='Temperature')
    plt.plot(monthly_data['month'], monthly_data['bikes_booked'], marker='o', linestyle='-', label='Bikes Booked')

    plt.title('Smoothed Monthly Mean Temperature and Bikes Booked (Normalized)')
    plt.xlabel('Month')
    plt.ylabel('Normalized Mean Value')
    plt.xticks(monthly_data['month'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}smoothed_monthly_mean_temp_vs_bikes_booked.png')
    plt.show()

# Example usage:
#plot_smoothed_monthly_mean_temp_vs_bikes_booked('path/to/data_file.csv', 'path/to/output_dir/')






# plot_monthly_mean_temp_vs_bikes_booked('path/to/data_file.csv', 'path/to/output_dir/')


#plot_normalized_daily_means_weather_data_with_bike_availibility('combined_citys/combined_city_data.csv')

file_path = 'Nürnberg/complete_nürnberg.csv'
plot_smoothed_monthly_mean_temp_vs_bikes_booked('combined_citys/combined_city_data.csv','',3)
#plot_average_bikes_per_station(file_path, '')
#plot_average_bikes_all_stations_combined(file_path, '')
#plot_normalized_bikes_and_bookings(file_path,'')
#plot_normalized_bikes_availability_bookings_and_returns(file_path,'')
#plot_average_bikes_and_counts(file_path, 'Nürnberg/bikes_nürnberg.csv', '')


#data_files = ['dresden/complete_dresden.csv', 'Essen/complete_essen.csv', 'heidelberg/complete_heidelberg.csv', 'Nürnberg/complete_nürnberg.csv']
#plot_average_bikes_all_stations_combined(data_files)