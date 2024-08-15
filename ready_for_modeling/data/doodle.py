import pandas as pd

def process_bike_data(input_csv, output_csv):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv)




    #Calculate bikes_difference and add it as a new column
    df['bikes_difference'] = df['bikes_returned'] - df['bikes_booked']
    

    # Write the updated DataFrame back to a new CSV file
    df.to_csv(output_csv, index=False)

# Example usage
input_csv = '2022_complete_essen.csv'  # Replace with your input CSV file path
output_csv = input_csv  # Replace with your desired output CSV file path
process_bike_data(input_csv, output_csv)
