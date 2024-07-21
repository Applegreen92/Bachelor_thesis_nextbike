import pandas as pd

target_file = 'bike_station_data_Essen.csv'
get_data_file = 'complete_weather_data_essen.csv'

target_df = pd.read_csv(target_file)
get_df = pd.read_csv(get_data_file)



merged_df = pd.merge(target_df, get_df, on=['datetime', 'station_name'], how='left')

merged_df.to_csv('complete_essen_weather_station_data.csv', index=False)