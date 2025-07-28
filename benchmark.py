import pandas as pd
import numpy as np
import os
from math import radians, cos, sin, asin, sqrt
from Ben_predictor.ben_predictor import predict_next_point as ben_predictor
from Yorgo_predictor.yorgo_predictor import predict_next_point as yorgo_predictor

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in meters
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in meters
    r = 6371000
    return c * r

def calculate_3d_distance(lat1, lon1, alt1, lat2, lon2, alt2):
    """
    Calculate 3D distance between two points including altitude difference
    Returns distance in meters
    """
    horizontal_dist = haversine_distance(lat1, lon1, lat2, lon2)
    vertical_dist = abs(alt2 - alt1)
    return sqrt(horizontal_dist**2 + vertical_dist**2)

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

def benchmark_predictor(predictor_func, predictor_name, df, n_points, key_columns=['gps_lat', 'gps_lon', 'gps_alt']):
    """
    Benchmark a single predictor function
    
    Args:
        predictor_func: The prediction function to test
        predictor_name: Name of the predictor for display
        df: DataFrame containing flight data
        n_points: Number of previous points to use for prediction
        key_columns: Columns to use for position (lat, lon, alt)
    
    Returns:
        dict: Results containing errors and statistics
    """
    results = {
        'predictor': predictor_name,
        'n_points': n_points,
        'predictions': [],
        'actual_values': [],
        'distances': [],
        'lat_errors': [],
        'lon_errors': [],
        'alt_errors': []
    }
    
    # Filter out rows with missing GPS data
    df_clean = df.dropna(subset=key_columns)
    
    if len(df_clean) < n_points + 1:
        print(f"Warning: Not enough data points for {predictor_name}. Need at least {n_points + 1}, got {len(df_clean)}")
        return results
    
    successful_predictions = 0
    failed_predictions = 0
    
    # Iterate through the dataset, using sliding window approach
    for i in range(n_points, len(df_clean)):
        try:
            # Get the last n_points for prediction
            input_data = df_clean.iloc[i-n_points:i].copy()
            
            # Get the actual next point
            actual_next = df_clean.iloc[i]
            actual_lat = actual_next[key_columns[0]]
            actual_lon = actual_next[key_columns[1]]
            actual_alt = actual_next[key_columns[2]]
            
            # Make prediction
            prediction = predictor_func(input_data)
            
            # Handle different return formats
            if prediction == -1 or prediction is None:
                failed_predictions += 1
                continue
            
            # Parse prediction based on format
            if isinstance(prediction, (list, tuple)) and len(prediction) >= 3:
                pred_lat, pred_lon, pred_alt = prediction[0], prediction[1], prediction[2]
            elif isinstance(prediction, dict):
                pred_lat = prediction.get('lat', prediction.get('latitude', 0))
                pred_lon = prediction.get('lon', prediction.get('longitude', 0))
                pred_alt = prediction.get('alt', prediction.get('altitude', 0))
            else:
                failed_predictions += 1
                continue
            
            # Calculate errors
            lat_error = abs(pred_lat - actual_lat)
            lon_error = abs(pred_lon - actual_lon)
            alt_error = abs(pred_alt - actual_alt)
            distance_error = calculate_3d_distance(pred_lat, pred_lon, pred_alt, 
                                                 actual_lat, actual_lon, actual_alt)
            
            # Store results
            results['predictions'].append([pred_lat, pred_lon, pred_alt])
            results['actual_values'].append([actual_lat, actual_lon, actual_alt])
            results['distances'].append(distance_error)
            results['lat_errors'].append(lat_error)
            results['lon_errors'].append(lon_error)
            results['alt_errors'].append(alt_error)
            
            successful_predictions += 1
            
        except Exception as e:
            failed_predictions += 1
            print(f"Error in prediction {i} for {predictor_name}: {e}")
            continue
    
    # Calculate statistics
    if results['distances']:
        results['mean_distance_error'] = np.mean(results['distances'])
        results['median_distance_error'] = np.median(results['distances'])
        results['std_distance_error'] = np.std(results['distances'])
        results['max_distance_error'] = np.max(results['distances'])
        results['min_distance_error'] = np.min(results['distances'])
        
        results['mean_lat_error'] = np.mean(results['lat_errors'])
        results['mean_lon_error'] = np.mean(results['lon_errors'])
        results['mean_alt_error'] = np.mean(results['alt_errors'])
        
        results['successful_predictions'] = successful_predictions
        results['failed_predictions'] = failed_predictions
        results['success_rate'] = successful_predictions / (successful_predictions + failed_predictions) * 100
    else:
        results['mean_distance_error'] = float('inf')
        results['successful_predictions'] = 0
        results['failed_predictions'] = failed_predictions
        results['success_rate'] = 0
    
    return results

def run_benchmark(n_points=5, datasets=None):
    """
    Run benchmark comparison between Ben and Yorgo predictors
    
    Args:
        n_points: Number of previous points to use for prediction
        datasets: List of dataset names to test on, or None for all
    
    Returns:
        dict: Complete benchmark results
    """
    print(f"Starting benchmark with {n_points} previous points...")
    print("=" * 60)
    
    # Load data
    df1, df2 = load_csv()
    if df1 is None or df2 is None:
        print("Failed to load data. Exiting benchmark.")
        return None
    
    datasets_dict = {
        'flight_log_2025-07-03': df1,
        'flight_log_2025-07-23': df2
    }
    
    if datasets:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in datasets}
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, df in datasets_dict.items():
        print(f"\nTesting on dataset: {dataset_name}")
        print("-" * 40)
        
        # Test Ben's predictor
        print(f"Testing Ben's predictor...")
        ben_results = benchmark_predictor(ben_predictor, "Ben", df, n_points)
        
        # Test Yorgo's predictor
        print(f"Testing Yorgo's predictor...")
        yorgo_results = benchmark_predictor(yorgo_predictor, "Yorgo", df, n_points)
        
        all_results[dataset_name] = {
            'ben': ben_results,
            'yorgo': yorgo_results
        }
        
        # Print results for this dataset
        print_dataset_results(ben_results, yorgo_results, dataset_name)
    
    # Print overall comparison
    print_overall_comparison(all_results)
    
    return all_results

def print_dataset_results(ben_results, yorgo_results, dataset_name):
    """Print results for a single dataset"""
    print(f"\nResults for {dataset_name}:")
    print("=" * 50)
    
    predictors = [("Ben", ben_results), ("Yorgo", yorgo_results)]
    
    for name, results in predictors:
        print(f"\n{name}'s Predictor:")
        print(f"  Success Rate: {results['success_rate']:.1f}% ({results['successful_predictions']}/{results['successful_predictions'] + results['failed_predictions']})")
        
        if results['successful_predictions'] > 0:
            print(f"  Mean 3D Distance Error: {results['mean_distance_error']:.2f} meters")
            print(f"  Median 3D Distance Error: {results['median_distance_error']:.2f} meters")
            print(f"  Std Dev Distance Error: {results['std_distance_error']:.2f} meters")
            print(f"  Max Distance Error: {results['max_distance_error']:.2f} meters")
            print(f"  Min Distance Error: {results['min_distance_error']:.2f} meters")
            print(f"  Mean Latitude Error: {results['mean_lat_error']:.6f} degrees")
            print(f"  Mean Longitude Error: {results['mean_lon_error']:.6f} degrees")
            print(f"  Mean Altitude Error: {results['mean_alt_error']:.2f} meters")
        else:
            print("  No successful predictions")

def print_overall_comparison(all_results):
    """Print overall comparison across all datasets"""
    print("\n" + "=" * 60)
    print("OVERALL COMPARISON")
    print("=" * 60)
    
    ben_total_success = 0
    ben_total_attempts = 0
    ben_all_distances = []
    
    yorgo_total_success = 0
    yorgo_total_attempts = 0
    yorgo_all_distances = []
    
    for dataset_name, results in all_results.items():
        ben_res = results['ben']
        yorgo_res = results['yorgo']
        
        ben_total_success += ben_res['successful_predictions']
        ben_total_attempts += ben_res['successful_predictions'] + ben_res['failed_predictions']
        ben_all_distances.extend(ben_res['distances'])
        
        yorgo_total_success += yorgo_res['successful_predictions']
        yorgo_total_attempts += yorgo_res['successful_predictions'] + yorgo_res['failed_predictions']
        yorgo_all_distances.extend(yorgo_res['distances'])
    
    print(f"\nOverall Success Rates:")
    ben_overall_rate = (ben_total_success / ben_total_attempts * 100) if ben_total_attempts > 0 else 0
    yorgo_overall_rate = (yorgo_total_success / yorgo_total_attempts * 100) if yorgo_total_attempts > 0 else 0
    
    print(f"  Ben: {ben_overall_rate:.1f}% ({ben_total_success}/{ben_total_attempts})")
    print(f"  Yorgo: {yorgo_overall_rate:.1f}% ({yorgo_total_success}/{yorgo_total_attempts})")
    
    if ben_all_distances and yorgo_all_distances:
        print(f"\nOverall Mean 3D Distance Errors:")
        print(f"  Ben: {np.mean(ben_all_distances):.2f} meters")
        print(f"  Yorgo: {np.mean(yorgo_all_distances):.2f} meters")
        
        print(f"\nOverall Median 3D Distance Errors:")
        print(f"  Ben: {np.median(ben_all_distances):.2f} meters")
        print(f"  Yorgo: {np.median(yorgo_all_distances):.2f} meters")
        
        # Determine winner
        ben_mean = np.mean(ben_all_distances)
        yorgo_mean = np.mean(yorgo_all_distances)
        
        if ben_mean < yorgo_mean:
            winner = "Ben"
            improvement = ((yorgo_mean - ben_mean) / yorgo_mean) * 100
        else:
            winner = "Yorgo"
            improvement = ((ben_mean - yorgo_mean) / ben_mean) * 100
        
        print(f"\nWINNER: {winner}'s predictor")
        print(f"Improvement: {improvement:.1f}% better average distance error")

def main():
    """Main function to run the benchmark"""
    print("Flight Path Predictor Benchmark")
    print("=" * 60)
    
    # Get user input for number of points
    try:
        n_points = int(input("Enter the number of previous points to use for prediction (default 5): ") or 5)
        if n_points < 1:
            print("Number of points must be at least 1. Using default value of 5.")
            n_points = 5
    except ValueError:
        print("Invalid input. Using default value of 5.")
        n_points = 5
    
    # Run benchmark
    results = run_benchmark(n_points)
    
    if results:
        print(f"\nBenchmark completed successfully!")
        print(f"Results stored in memory for further analysis if needed.")
    else:
        print("Benchmark failed to complete.")

if __name__ == "__main__":
    main()