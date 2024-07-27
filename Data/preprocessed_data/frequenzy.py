import pandas as pd
import matplotlib.pyplot as plt

def plot_average_bikes_all_stations_combined(data_file, output_dir, filter_weekday=None):
    """
    Plots the average bike availability for all stations combined per hour over the year.

    Parameters:
    data_file (str): Path to the CSV file containing the data.
    output_dir (str): Directory where the plots will be saved.
    filter_weekday (str): 'weekday', 'weekend', or None to filter the data. Default is None.
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Convert the datetime column to a datetime object
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Filter data if needed
    if filter_weekday == 'weekday':
        df = df[df['is_weekend'] == 0]
    elif filter_weekday == 'weekend':
        df = df[df['is_weekend'] == 1]

    # Extract the hour and minute components as a string format for easier plotting
    df['time'] = df['datetime'].dt.strftime('%H:%M')

    # Group by time to calculate the mean bike availability across all stations
    hourly_availability = df.groupby('time')['bikes_available'].mean().reset_index()

    # Plot the average bikes available per hour for all stations combined
    plt.figure(figsize=(18, 6))
    plt.plot(hourly_availability['time'], hourly_availability['bikes_available'], marker='o', linestyle='-')
    plt.title('Average Bikes Available per Hour for All Stations')
    if filter_weekday:
        plt.title(f'Average Bikes Available per Hour for All Stations ({filter_weekday.capitalize()})')
    plt.xlabel('Time of the Day')
    plt.ylabel('Average Bikes Available')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}average_bikes_available_per_hour_all_stations_combined_{filter_weekday}.png')
    plt.show()

file_path = 'Checked_preprocessed_data/combined_citys/combined_city_data.csv'
plot_average_bikes_all_stations_combined(file_path, '', filter_weekday='weekday')
plot_average_bikes_all_stations_combined(file_path, '', filter_weekday='weekend')
plot_average_bikes_all_stations_combined(file_path, '')
