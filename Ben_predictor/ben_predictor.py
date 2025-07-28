import pandas as pd
import os
import numpy as np

def load_csv():
    """
    Load CSV files from the parsed_data folder and return them as DataFrames.
    
    Returns:
        tuple: (df1, df2) - Two pandas DataFrames containing the flight log data
    """
    # Define the path to the parsed data folder
    data_folder = os.path.join(os.path.dirname(__file__), 'parsed_data')

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
    """
    Predict the next GPS point based on average velocity and direction from last two points.
    
    Args:
        df: DataFrame containing the last N GPS points with columns 'gps_lat', 'gps_lon', 'gps_alt'
    
    Returns:
        tuple: (predicted_lat, predicted_lon, predicted_alt)
    """
    try:
        # Check if we have sufficient data
        if len(df) < 2:
            return -1, -1, -1
        
        # Extract GPS coordinates
        lats = df['gps_lat'].values
        lons = df['gps_lon'].values
        alts = df['gps_alt'].values
        
        # Check for valid data
        if np.any(np.isnan(lats)) or np.any(np.isnan(lons)) or np.any(np.isnan(alts)):
            return -1, -1, -1
        
        # Calculate time intervals (assuming roughly equal time intervals)
        # Since we don't have precise timestamps, we'll assume unit time intervals
        n_points = len(df)
        
        # Calculate velocity components for each interval
        lat_velocities = []
        lon_velocities = []
        alt_velocities = []
        
        for i in range(1, n_points):
            lat_vel = lats[i] - lats[i-1]
            lon_vel = lons[i] - lons[i-1]
            alt_vel = alts[i] - alts[i-1]
            
            lat_velocities.append(lat_vel)
            lon_velocities.append(lon_vel)
            alt_velocities.append(alt_vel)
        
        # Calculate average velocities
        avg_lat_vel = np.mean(lat_velocities)
        avg_lon_vel = np.mean(lon_velocities)
        avg_alt_vel = np.mean(alt_velocities)
        
        # Get direction from the last two points (most recent trend)
        if n_points >= 2:
            # Direction vector from second-to-last to last point
            last_lat_dir = lats[-1] - lats[-2]
            last_lon_dir = lons[-1] - lons[-2]
            last_alt_dir = alts[-1] - alts[-2]
            
            # Combine average velocity with recent direction
            # Weight recent direction more heavily (70%) vs average velocity (30%)
            direction_weight = 0.7
            velocity_weight = 0.3
            
            predicted_lat_change = (direction_weight * last_lat_dir) + (velocity_weight * avg_lat_vel)
            predicted_lon_change = (direction_weight * last_lon_dir) + (velocity_weight * avg_lon_vel)
            predicted_alt_change = (direction_weight * last_alt_dir) + (velocity_weight * avg_alt_vel)
        else:
            # Fallback to just average velocity
            predicted_lat_change = avg_lat_vel
            predicted_lon_change = avg_lon_vel
            predicted_alt_change = avg_alt_vel
        
        # Predict next point
        predicted_lat = lats[-1] + predicted_lat_change
        predicted_lon = lons[-1] + predicted_lon_change
        predicted_alt = alts[-1] + predicted_alt_change
        
        # Ensure altitude is non-negative
        predicted_alt = max(0, predicted_alt)
        
        return predicted_lat, predicted_lon, predicted_alt
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, -1, -1