import pandas as pd

target_file = '../Nürnberg/bikes_nürnberg.csv'
get_data_file = 'weather_data_essen.csv'

target_df = pd.read_csv(target_file)
get_df = pd.read_csv(get_data_file)



merged_df = pd.merge(target_df, get_df, on=['datetime', 'station_name'], how='left')

merged_df.to_csv('complete_essen.csv', index=False)