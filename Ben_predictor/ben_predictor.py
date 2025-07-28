import pandas as pd
import os
import numpy as np


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
        
        n_points = len(df)
        
        # Handle time intervals - check if gps_time column exists
        use_timestamps = False
        time_diffs = None
        
        if 'gps_time' in df.columns:
            try:
                # Check for NaN values in gps_time first
                gps_times = df['gps_time'].values
                if np.any(np.isnan(gps_times)):
                    print("Warning: NaN values found in gps_time, falling back to unit intervals")
                else:
                    # gps_time contains Unix timestamps - convert to pandas datetime
                    timestamps = pd.to_datetime(df['gps_time'], unit='s')
                    time_diffs = timestamps.diff().dt.total_seconds().values[1:]  # Skip first NaN
                    
                    # Debug: print some info about time differences
                    if len(time_diffs) > 0:
                        valid_diffs = time_diffs[~np.isnan(time_diffs)]
                        # Filter out zero time differences (duplicate timestamps)
                        positive_diffs = valid_diffs[valid_diffs > 0]
                        
                        if len(positive_diffs) > 0:
                            # Use only positive time differences
                            use_timestamps = True
                            # Replace zero differences with interpolated values
                            processed_time_diffs = []
                            avg_positive_diff = np.mean(positive_diffs)
                            
                            for diff in time_diffs:
                                if np.isnan(diff) or diff <= 0:
                                    # Replace invalid/zero differences with average positive difference
                                    processed_time_diffs.append(avg_positive_diff)
                                else:
                                    processed_time_diffs.append(diff)
                            
                            time_diffs = np.array(processed_time_diffs)
                            print(f"Info: Using timestamps with {len(positive_diffs)}/{len(valid_diffs)} valid intervals, avg: {avg_positive_diff:.1f}s")
                        else:
                            print(f"Debug: All time differences are zero or negative")
                            print("Warning: No positive time differences found, falling back to unit intervals")
                    else:
                        print("Warning: No time differences calculated, falling back to unit intervals")
            except Exception as e:
                print(f"Warning: Could not process gps_time ({e}), using unit intervals")
        
        # Calculate velocity components for each interval
        lat_velocities = []
        lon_velocities = []
        alt_velocities = []
        
        for i in range(1, n_points):
            lat_change = lats[i] - lats[i-1]
            lon_change = lons[i] - lons[i-1]
            alt_change = alts[i] - alts[i-1]
            
            if use_timestamps and i-1 < len(time_diffs):
                # Calculate actual velocity per second
                time_interval = time_diffs[i-1]
                lat_vel = lat_change / time_interval
                lon_vel = lon_change / time_interval
                alt_vel = alt_change / time_interval
            else:
                # Use unit time intervals (original behavior)
                lat_vel = lat_change
                lon_vel = lon_change
                alt_vel = alt_change
            
            lat_velocities.append(lat_vel)
            lon_velocities.append(lon_vel)
            alt_velocities.append(alt_vel)
        
        # Calculate average velocities
        avg_lat_vel = np.mean(lat_velocities)
        avg_lon_vel = np.mean(lon_velocities)
        avg_alt_vel = np.mean(alt_velocities)
        
        # Get direction from the last two points (most recent trend)
        if n_points >= 2:
            # Get the most recent time interval for prediction
            prediction_time_interval = 1.0  # Default to 1 second
            if use_timestamps and len(time_diffs) > 0:
                # Use the average of recent time intervals for prediction
                recent_intervals = time_diffs[-min(3, len(time_diffs)):]  # Last 3 intervals or available
                prediction_time_interval = np.mean(recent_intervals)
            
            # Direction vector from second-to-last to last point
            if use_timestamps and len(time_diffs) > 0:
                # For timestamped data, use actual velocity-based direction
                last_lat_dir = lat_velocities[-1] * prediction_time_interval
                last_lon_dir = lon_velocities[-1] * prediction_time_interval
                last_alt_dir = alt_velocities[-1] * prediction_time_interval
            else:
                # For unit intervals, use position differences
                last_lat_dir = lats[-1] - lats[-2]
                last_lon_dir = lons[-1] - lons[-2]
                last_alt_dir = alts[-1] - alts[-2]
            
            # Calculate average velocity changes for prediction
            if use_timestamps:
                # For timestamped data, velocities are per second, scale by prediction interval
                avg_lat_change = avg_lat_vel * prediction_time_interval
                avg_lon_change = avg_lon_vel * prediction_time_interval
                avg_alt_change = avg_alt_vel * prediction_time_interval
            else:
                # For unit intervals, use velocities directly
                avg_lat_change = avg_lat_vel
                avg_lon_change = avg_lon_vel
                avg_alt_change = avg_alt_vel
            
            # Combine average velocity with recent direction
            # Weight recent direction more heavily (70%) vs average velocity (30%)
            direction_weight = 0.7
            velocity_weight = 0.3
            
            predicted_lat_change = (direction_weight * last_lat_dir) + (velocity_weight * avg_lat_change)
            predicted_lon_change = (direction_weight * last_lon_dir) + (velocity_weight * avg_lon_change)
            predicted_alt_change = (direction_weight * last_alt_dir) + (velocity_weight * avg_alt_change)
        else:
            # Fallback to just average velocity
            if use_timestamps and len(time_diffs) > 0:
                prediction_time_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 1.0
                predicted_lat_change = avg_lat_vel * prediction_time_interval
                predicted_lon_change = avg_lon_vel * prediction_time_interval
                predicted_alt_change = avg_alt_vel * prediction_time_interval
            else:
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