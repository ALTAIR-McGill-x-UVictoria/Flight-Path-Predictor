import pandas as pd
import os

def load_csv():
    """
    Load CSV files from the parsed_data folder and return them as DataFrames.
    
    Returns:
        tuple: (df1, df2) - Two pandas DataFrames containing the flight log data
    """
    # Define the path to the parsed data folder
    data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'parsed_data')

    # Define the CSV file paths
    csv_file1 = os.path.join(data_folder, 'flight_log_2025-07-03_22-10-39_parsed.csv')
    csv_file2 = os.path.join(data_folder, 'flight_log_2025-07-23_23-12-23_parsed.csv')

    # Read the CSV files into pandas DataFrames
    try:
        df1 = pd.read_csv(csv_file1)
        print(f"Successfully loaded {csv_file1}")
        print(f"Shape: {df1.shape}")
        print(f"Columns: {list(df1.columns)}")
        print()
        
        df2 = pd.read_csv(csv_file2)
        print(f"Successfully loaded {csv_file2}")
        print(f"Shape: {df2.shape}")
        print(f"Columns: {list(df2.columns)}")
        print()
        
        # Display first few rows of each dataset
        print("First 5 rows of flight_log_2025-07-03:")
        print(df1.head())
        print()
        
        print("First 5 rows of flight_log_2025-07-23:")
        print(df2.head())
        
        return df1, df2
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file - {e}")
        return None, None
    except Exception as e:
        print(f"Error reading CSV files: {e}")
        return None, None


# write code in this function to predict next point
def predict_next_point(df):
    predicted_point = -1

    return predicted_point