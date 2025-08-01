import pandas as pd
import numpy as np
import os
import sys
from math import radians, cos, sin, asin, sqrt
from Ben_predictor.ben_predictor import predict_next_point as ben_predictor
from Yorgo_predictor.yorgo_predictor import predict_next_point as yorgo_predictor

# Optional imports for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

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
    csv_file2 = os.path.join(data_folder, 'flight_log_2025-07-23_21-41-08_parsed.csv')

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
        'alt_errors': [],
        'altitudes': []  # Track actual altitudes for apogee detection
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
                
                # Check for error return values (-1, -1, -1)
                if pred_lat == -1 and pred_lon == -1 and pred_alt == -1:
                    failed_predictions += 1
                    continue
                    
                # Additional validation for reasonable coordinate values
                if (abs(pred_lat) > 90 or abs(pred_lon) > 180 or 
                    pred_alt < -500 or pred_alt > 50000 or
                    np.isnan(pred_lat) or np.isnan(pred_lon) or np.isnan(pred_alt)):
                    failed_predictions += 1
                    continue
                    
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
            
            # Final sanity check - reject predictions with impossible distances (> 100km)
            if distance_error > 100000:  # 100km threshold for reasonable balloon movement
                failed_predictions += 1
                print(f"Warning: Rejected prediction with distance error {distance_error:.0f}m at index {i}")
                continue
            
            # Store results
            results['predictions'].append([pred_lat, pred_lon, pred_alt])
            results['actual_values'].append([actual_lat, actual_lon, actual_alt])
            results['distances'].append(distance_error)
            results['lat_errors'].append(lat_error)
            results['lon_errors'].append(lon_error)
            results['alt_errors'].append(alt_error)
            results['altitudes'].append(actual_alt)  # Track altitude for apogee detection
            
            successful_predictions += 1
            
        except Exception as e:
            failed_predictions += 1
            print(f"Error in prediction {i} for {predictor_name}: {e}")
            continue
    
    # Calculate statistics with outlier handling
    if results['distances']:
        distances_array = np.array(results['distances'])
        
        # Calculate percentiles to identify outliers
        p95 = np.percentile(distances_array, 95)
        p99 = np.percentile(distances_array, 99)
        
        # Create filtered datasets for more meaningful statistics
        valid_predictions = distances_array[distances_array < 1000]  # Under 1km (reasonable for balloon)
        excellent_predictions = distances_array[distances_array < 100]  # Under 100m (excellent)
        
        results['mean_distance_error'] = np.mean(distances_array)
        results['median_distance_error'] = np.median(distances_array)
        results['std_distance_error'] = np.std(distances_array)
        results['max_distance_error'] = np.max(distances_array)
        results['min_distance_error'] = np.min(distances_array)
        
        # Add robust statistics
        results['p95_distance_error'] = p95
        results['p99_distance_error'] = p99
        results['trimmed_mean_distance_error'] = np.mean(valid_predictions) if len(valid_predictions) > 0 else float('inf')
        results['excellent_prediction_rate'] = len(excellent_predictions) / len(distances_array) * 100
        results['valid_prediction_rate'] = len(valid_predictions) / len(distances_array) * 100
        
        # Count outliers
        outliers_1km = np.sum(distances_array >= 1000)
        outliers_10km = np.sum(distances_array >= 10000) 
        results['outliers_over_1km'] = outliers_1km
        results['outliers_over_10km'] = outliers_10km
        
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

def run_benchmark(n_points=5, datasets=None, predictors=None):
    """
    Run benchmark comparison between Ben and Yorgo predictors
    
    Args:
        n_points: Number of previous points to use for prediction
        datasets: List of dataset names to test on, or None for all
        predictors: List of predictor names to test ('ben', 'yorgo'), or None for all
    
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
    
    print(f"\nNote: Both predictor functions currently return -1 (placeholder implementation).")
    print(f"This benchmark will show 0% success rate until actual prediction algorithms are implemented.")
    print(f"The framework is ready to test real implementations when available.\n")
    
    datasets_dict = {
        'flight_log_2025-07-03': df1,
        'flight_log_2025-07-23': df2
    }
    
    if datasets:
        datasets_dict = {k: v for k, v in datasets_dict.items() if k in datasets}
    
    # Set default predictors if none specified
    if predictors is None:
        predictors = ['ben', 'yorgo']
    
    all_results = {}
    
    # Test each dataset
    for dataset_name, df in datasets_dict.items():
        print(f"\nTesting on dataset: {dataset_name}")
        print("-" * 40)
        
        dataset_results = {}
        
        # Test Ben's predictor if requested
        if 'ben' in predictors:
            print(f"Testing Ben's predictor...")
            ben_results = benchmark_predictor(ben_predictor, "Ben", df, n_points)
            dataset_results['ben'] = ben_results
        
        # Test Yorgo's predictor if requested
        if 'yorgo' in predictors:
            print(f"Testing Yorgo's predictor...")
            yorgo_results = benchmark_predictor(yorgo_predictor, "Yorgo", df, n_points)
            dataset_results['yorgo'] = yorgo_results
        
        all_results[dataset_name] = dataset_results
        
        # Print results for this dataset
        ben_res = dataset_results.get('ben')
        yorgo_res = dataset_results.get('yorgo')
        print_dataset_results(ben_res, yorgo_res, dataset_name)
    
    # Print overall comparison
    print_overall_comparison(all_results)
    
    # Create visualizations if matplotlib is available
    if MATPLOTLIB_AVAILABLE:
        print("\nGenerating accuracy visualizations...")
        plot_accuracy_comparison(all_results)
        plot_error_trends(all_results)
    else:
        print("\nSkipping visualizations (matplotlib not available)")
    
    # Create and display summary table
    summary_table = create_summary_table(all_results)
    print("\nSummary Table:")
    print("=" * 120)
    print(summary_table.to_string(index=False))
    
    return all_results

def print_dataset_results(ben_results, yorgo_results, dataset_name):
    """Print results for a single dataset"""
    print(f"\nResults for {dataset_name}:")
    print("=" * 50)
    
    predictors = []
    if ben_results is not None:
        predictors.append(("Ben", ben_results))
    if yorgo_results is not None:
        predictors.append(("Yorgo", yorgo_results))
    
    for name, results in predictors:
        print(f"\n{name}'s Predictor:")
        print(f"  Success Rate: {results['success_rate']:.1f}% ({results['successful_predictions']}/{results['successful_predictions'] + results['failed_predictions']})")
        
        if results['successful_predictions'] > 0:
            print(f"  Mean 3D Distance Error: {results['mean_distance_error']:.2f} meters")
            print(f"  Median 3D Distance Error: {results['median_distance_error']:.2f} meters")
            print(f"  95th Percentile Error: {results['p95_distance_error']:.2f} meters")
            print(f"  Trimmed Mean (< 1km): {results['trimmed_mean_distance_error']:.2f} meters")
            print(f"  Std Dev Distance Error: {results['std_distance_error']:.2f} meters")
            print(f"  Max Distance Error: {results['max_distance_error']:.2f} meters")
            print(f"  Min Distance Error: {results['min_distance_error']:.2f} meters")
            print(f"  Excellent Predictions (< 100m): {results['excellent_prediction_rate']:.1f}%")
            print(f"  Valid Predictions (< 1km): {results['valid_prediction_rate']:.1f}%")
            if results['outliers_over_1km'] > 0:
                print(f"  ⚠️  Outliers > 1km: {results['outliers_over_1km']} predictions")
            if results['outliers_over_10km'] > 0:
                print(f"  ⚠️  Major outliers > 10km: {results['outliers_over_10km']} predictions")
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
    
    # Check which predictors we have results for
    has_ben = False
    has_yorgo = False
    
    for dataset_name, results in all_results.items():
        if 'ben' in results and results['ben'] is not None:
            ben_res = results['ben']
            ben_total_success += ben_res['successful_predictions']
            ben_total_attempts += ben_res['successful_predictions'] + ben_res['failed_predictions']
            ben_all_distances.extend(ben_res['distances'])
            has_ben = True
        
        if 'yorgo' in results and results['yorgo'] is not None:
            yorgo_res = results['yorgo']
            yorgo_total_success += yorgo_res['successful_predictions']
            yorgo_total_attempts += yorgo_res['successful_predictions'] + yorgo_res['failed_predictions']
            yorgo_all_distances.extend(yorgo_res['distances'])
            has_yorgo = True
    
    print(f"\nOverall Success Rates:")
    if has_ben:
        ben_overall_rate = (ben_total_success / ben_total_attempts * 100) if ben_total_attempts > 0 else 0
        print(f"  Ben: {ben_overall_rate:.1f}% ({ben_total_success}/{ben_total_attempts})")
    
    if has_yorgo:
        yorgo_overall_rate = (yorgo_total_success / yorgo_total_attempts * 100) if yorgo_total_attempts > 0 else 0
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
    elif ben_all_distances:
        print(f"\nBen's Performance:")
        print(f"  Mean 3D Distance Error: {np.mean(ben_all_distances):.2f} meters")
        print(f"  Median 3D Distance Error: {np.median(ben_all_distances):.2f} meters")
    elif yorgo_all_distances:
        print(f"\nYorgo's Performance:")
        print(f"  Mean 3D Distance Error: {np.mean(yorgo_all_distances):.2f} meters")
        print(f"  Median 3D Distance Error: {np.median(yorgo_all_distances):.2f} meters")

def plot_accuracy_comparison(all_results, save_plots=True):
    """
    Create visualizations comparing the accuracy of both predictors
    
    Args:
        all_results: Dictionary containing benchmark results
        save_plots: Whether to save plots to files
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping accuracy comparison plots.")
        return
        
    # Set plotting style
    if SEABORN_AVAILABLE:
        try:
            plt.style.use('seaborn-v0_8')
        except:
            try:
                plt.style.use('seaborn')
            except:
                plt.style.use('default')
    else:
        plt.style.use('default')
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Flight Path Predictor Accuracy Comparison', fontsize=16, fontweight='bold')
    
    # Collect data for all datasets
    ben_distances = []
    yorgo_distances = []
    ben_lat_errors = []
    yorgo_lat_errors = []
    ben_lon_errors = []
    yorgo_lon_errors = []
    ben_alt_errors = []
    yorgo_alt_errors = []
    dataset_labels = []
    
    for dataset_name, results in all_results.items():
        if 'ben' in results and results['ben'] is not None:
            ben_res = results['ben']
            ben_distances.extend(ben_res['distances'])
            ben_lat_errors.extend(ben_res['lat_errors'])
            ben_lon_errors.extend(ben_res['lon_errors'])
            ben_alt_errors.extend(ben_res['alt_errors'])
            # Add dataset labels for each distance measurement
            dataset_labels.extend([dataset_name] * len(ben_res['distances']))
        
        if 'yorgo' in results and results['yorgo'] is not None:
            yorgo_res = results['yorgo']
            yorgo_distances.extend(yorgo_res['distances'])
            yorgo_lat_errors.extend(yorgo_res['lat_errors'])
            yorgo_lon_errors.extend(yorgo_res['lon_errors'])
            yorgo_alt_errors.extend(yorgo_res['alt_errors'])
    
    # Use all data points without filtering outliers
    ben_distances_filtered = ben_distances
    yorgo_distances_filtered = yorgo_distances
    
    # No outliers hidden since we're showing all data
    ben_outliers = 0
    yorgo_outliers = 0
    
    # Plot 1: Distance Error Distribution (Box Plot)
    ax1 = axes[0, 0]
    if ben_distances_filtered and yorgo_distances_filtered:
        distance_data = [ben_distances_filtered, yorgo_distances_filtered]
        box_plot = ax1.boxplot(distance_data, tick_labels=['Ben', 'Yorgo'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
        box_plot['boxes'][1].set_facecolor('lightcoral')
    elif ben_distances_filtered:
        # Only Ben has data
        box_plot = ax1.boxplot([ben_distances_filtered], tick_labels=['Ben'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightblue')
    elif yorgo_distances_filtered:
        # Only Yorgo has data
        box_plot = ax1.boxplot([yorgo_distances_filtered], tick_labels=['Yorgo'], patch_artist=True)
        box_plot['boxes'][0].set_facecolor('lightcoral')
    else:
        ax1.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax1.transAxes)
    ax1.set_title('3D Distance Error Distribution')
    ax1.set_ylabel('Distance Error (meters)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Distance Error Histogram
    ax2 = axes[0, 1]
    if ben_distances_filtered and yorgo_distances_filtered:
        ax2.hist(ben_distances_filtered, bins=30, alpha=0.7, label='Ben', color='lightblue', density=True)
        ax2.hist(yorgo_distances_filtered, bins=30, alpha=0.7, label='Yorgo', color='lightcoral', density=True)
        ax2.set_title('Distance Error Distribution')
        ax2.set_xlabel('Distance Error (meters)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    elif ben_distances_filtered:
        ax2.hist(ben_distances_filtered, bins=30, alpha=0.7, label='Ben', color='lightblue', density=True)
        ax2.set_title('Distance Error Distribution')
        ax2.set_xlabel('Distance Error (meters)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    elif yorgo_distances_filtered:
        ax2.hist(yorgo_distances_filtered, bins=30, alpha=0.7, label='Yorgo', color='lightcoral', density=True)
        ax2.set_title('Distance Error Distribution')
        ax2.set_xlabel('Distance Error (meters)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success Rate Comparison
    ax3 = axes[0, 2]
    success_rates = []
    predictor_names = []
    colors = []
    
    for dataset_name, results in all_results.items():
        if 'ben' in results and results['ben'] is not None:
            ben_rate = results['ben']['success_rate']
            success_rates.append(ben_rate)
            predictor_names.append('Ben')
            colors.append('lightblue')
        
        if 'yorgo' in results and results['yorgo'] is not None:
            yorgo_rate = results['yorgo']['success_rate']
            success_rates.append(yorgo_rate)
            predictor_names.append('Yorgo')
            colors.append('lightcoral')
    
    bars = ax3.bar(range(len(success_rates)), success_rates, color=colors)
    ax3.set_title('Success Rate Comparison')
    ax3.set_ylabel('Success Rate (%)')
    ax3.set_xlabel('Predictor')
    ax3.set_xticks(range(len(success_rates)))
    ax3.set_xticklabels(predictor_names)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom')
    
    # Plot 4: Latitude Error Comparison
    ax4 = axes[1, 0]
    if ben_lat_errors and yorgo_lat_errors:
        lat_data = [ben_lat_errors, yorgo_lat_errors]
        box_plot_lat = ax4.boxplot(lat_data, tick_labels=['Ben', 'Yorgo'], patch_artist=True)
        box_plot_lat['boxes'][0].set_facecolor('lightblue')
        box_plot_lat['boxes'][1].set_facecolor('lightcoral')
    else:
        ax4.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax4.transAxes)
    ax4.set_title('Latitude Error Distribution')
    ax4.set_ylabel('Latitude Error (degrees)')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Longitude Error Comparison
    ax5 = axes[1, 1]
    if ben_lon_errors and yorgo_lon_errors:
        lon_data = [ben_lon_errors, yorgo_lon_errors]
        box_plot_lon = ax5.boxplot(lon_data, tick_labels=['Ben', 'Yorgo'], patch_artist=True)
        box_plot_lon['boxes'][0].set_facecolor('lightblue')
        box_plot_lon['boxes'][1].set_facecolor('lightcoral')
    else:
        ax5.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax5.transAxes)
    ax5.set_title('Longitude Error Distribution')
    ax5.set_ylabel('Longitude Error (degrees)')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Altitude Error Comparison
    ax6 = axes[1, 2]
    if ben_alt_errors and yorgo_alt_errors:
        alt_data = [ben_alt_errors, yorgo_alt_errors]
        box_plot_alt = ax6.boxplot(alt_data, tick_labels=['Ben', 'Yorgo'], patch_artist=True)
        box_plot_alt['boxes'][0].set_facecolor('lightblue')
        box_plot_alt['boxes'][1].set_facecolor('lightcoral')
    else:
        ax6.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax6.transAxes)
    ax6.set_title('Altitude Error Distribution')
    ax6.set_ylabel('Altitude Error (meters)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('flight_predictor_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\nPlot saved as 'flight_predictor_accuracy_comparison.png'")
    
    plt.show()

def detect_apogee(altitudes):
    """
    Detect the apogee (maximum altitude) point in a flight
    
    Args:
        altitudes: List of altitude values
    
    Returns:
        int: Index of the apogee point, or -1 if no valid apogee found
    """
    if not altitudes or len(altitudes) < 3:
        return -1
    
    # Find the maximum altitude point
    max_alt = max(altitudes)
    apogee_candidates = [i for i, alt in enumerate(altitudes) if alt == max_alt]
    
    # If there are multiple points with max altitude, take the middle one
    if apogee_candidates:
        return apogee_candidates[len(apogee_candidates) // 2]
    
    return -1

def plot_error_trends(all_results, save_plots=True):
    """
    Create plots showing error trends over time for each dataset
    
    Args:
        all_results: Dictionary containing benchmark results
        save_plots: Whether to save plots to files
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Skipping error trend plots.")
        return
        
    # Load original CSV data to detect actual flight apogee
    df1, df2 = load_csv()
    csv_datasets = {
        'flight_log_2025-07-03': df1,
        'flight_log_2025-07-23': df2
    }
        
    n_datasets = len(all_results)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(15, 6 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Error Trends Over Flight Time (with Flight Apogee Markers)', fontsize=16, fontweight='bold')
    
    for idx, (dataset_name, results) in enumerate(all_results.items()):
        ben_res = results.get('ben')
        yorgo_res = results.get('yorgo')
        
        # Collect all distance data for outlier analysis
        all_distances = []
        if ben_res and ben_res['distances']:
            all_distances.extend(ben_res['distances'])
        if yorgo_res and yorgo_res['distances']:
            all_distances.extend(yorgo_res['distances'])
        
        # Calculate outlier threshold for reference but don't filter data
        if all_distances:
            q95 = np.percentile(all_distances, 95)
            mean_dist = np.mean(all_distances)
            std_dist = np.std(all_distances)
            outlier_threshold = min(q95, mean_dist + 3 * std_dist)
            # Set a reasonable maximum threshold for y-axis scaling
            outlier_threshold = max(outlier_threshold, 100)  # At least 100m for visibility
        else:
            outlier_threshold = 100  # Default threshold
        
        # Plot distance errors over time
        ax1 = axes[idx, 0]
        has_data = False
        
        # Plot Ben's results if available - show all data points
        if ben_res and ben_res['distances']:
            ben_distances = np.array(ben_res['distances'])
            x_ben = range(len(ben_distances))
            
            # Plot all points without filtering
            ax1.plot(x_ben, ben_distances, 'b-', alpha=0.7, label='Ben', linewidth=1)
            
            # Add moving average for trend visibility
            if len(ben_distances) > 10:
                # Create moving average
                ben_smooth = []
                for i in range(len(ben_distances)):
                    window_start = max(0, i-5)
                    window_end = min(len(ben_distances), i+6)
                    window_data = ben_distances[window_start:window_end]
                    ben_smooth.append(np.mean(window_data))
                ax1.plot(x_ben, ben_smooth, 'b-', linewidth=2, alpha=0.8, label='Ben (smoothed)')
            has_data = True
        
        # Plot Yorgo's results if available - show all data points
        if yorgo_res and yorgo_res['distances']:
            yorgo_distances = np.array(yorgo_res['distances'])
            x_yorgo = range(len(yorgo_distances))
            
            # Plot all points without filtering
            ax1.plot(x_yorgo, yorgo_distances, 'r-', alpha=0.7, label='Yorgo', linewidth=1)
            
            # Add moving average for trend visibility
            if len(yorgo_distances) > 10:
                # Create moving average
                yorgo_smooth = []
                for i in range(len(yorgo_distances)):
                    window_start = max(0, i-5)
                    window_end = min(len(yorgo_distances), i+6)
                    window_data = yorgo_distances[window_start:window_end]
                    yorgo_smooth.append(np.mean(window_data))
                ax1.plot(x_yorgo, yorgo_smooth, 'r-', linewidth=2, alpha=0.8, label='Yorgo (smoothed)')
            has_data = True
        
        # Add apogee markers for original CSV flight data
        if dataset_name in csv_datasets and csv_datasets[dataset_name] is not None:
            df = csv_datasets[dataset_name]
            # Clean the data and get altitudes
            key_columns = ['gps_lat', 'gps_lon', 'gps_alt']
            df_clean = df.dropna(subset=key_columns)
            
            if len(df_clean) > 0:
                # Get altitude data from the original CSV
                csv_altitudes = df_clean['gps_alt'].tolist()
                csv_apogee_idx = detect_apogee(csv_altitudes)
                
                if csv_apogee_idx >= 0:
                    # Since predictions start from n_points index, we need to adjust
                    # the apogee index to match the prediction timeline
                    n_points = 5  # Default value used in benchmark
                    if csv_apogee_idx >= n_points:
                        adjusted_apogee_idx = csv_apogee_idx - n_points
                        max_alt = csv_altitudes[csv_apogee_idx]
                        
                        # Only show marker if it's within the prediction range
                        max_predictions = 0
                        if ben_res and ben_res['distances']:
                            max_predictions = max(max_predictions, len(ben_res['distances']))
                        if yorgo_res and yorgo_res['distances']:
                            max_predictions = max(max_predictions, len(yorgo_res['distances']))
                        
                        if adjusted_apogee_idx < max_predictions:
                            # Add vertical line for flight apogee
                            ax1.axvline(x=adjusted_apogee_idx, color='green', linestyle=':', 
                                       alpha=0.8, linewidth=3, label=f'Flight Apogee ({max_alt:.0f}m)')
                            
                            # Add annotation for flight apogee
                            y_annotation = outlier_threshold * 0.7
                            ax1.annotate(f'Flight Apogee\n{max_alt:.0f}m', 
                                        xy=(adjusted_apogee_idx, y_annotation), 
                                        xytext=(adjusted_apogee_idx + max_predictions * 0.03, y_annotation * 1.2),
                                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.8),
                                        fontsize=10, color='green', ha='left', fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
        
        # Mark apogee (maximum altitude) points for predictor results
        if ben_res and ben_res.get('altitudes'):
            apogee_idx = detect_apogee(ben_res['altitudes'])
            if apogee_idx >= 0 and apogee_idx < len(ben_res['distances']):
                # Show apogee marker for all valid data points
                ax1.axvline(x=apogee_idx, color='blue', linestyle='--', alpha=0.6, linewidth=2, 
                           label='Ben Prediction Apogee')
                # Add annotation
                y_pos = ben_res['distances'][apogee_idx]
                ax1.annotate(f'Ben Pred.\nApogee\n({ben_res["altitudes"][apogee_idx]:.0f}m)', 
                            xy=(apogee_idx, y_pos), xytext=(apogee_idx + len(ben_res['distances']) * 0.05, y_pos * 0.8),
                            arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                            fontsize=8, color='blue', ha='left')
        
        if yorgo_res and yorgo_res.get('altitudes'):
            apogee_idx = detect_apogee(yorgo_res['altitudes'])
            if apogee_idx >= 0 and apogee_idx < len(yorgo_res['distances']):
                # Show apogee marker for all valid data points
                ax1.axvline(x=apogee_idx, color='red', linestyle='--', alpha=0.6, linewidth=2, 
                           label='Yorgo Prediction Apogee')
                # Add annotation
                y_pos = yorgo_res['distances'][apogee_idx]
                ax1.annotate(f'Yorgo Pred.\nApogee\n({yorgo_res["altitudes"][apogee_idx]:.0f}m)', 
                            xy=(apogee_idx, y_pos), xytext=(apogee_idx + len(yorgo_res['distances']) * 0.05, y_pos * 0.8),
                            arrowprops=dict(arrowstyle='->', color='red', alpha=0.6),
                            fontsize=8, color='red', ha='left')
        
        if has_data:
            ax1.legend()
            # Use dynamic y-axis limits to show all data including outliers
            if all_distances:
                max_distance = max(all_distances)
                ax1.set_ylim(0, max_distance * 1.1)
        else:
            ax1.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax1.transAxes)
        
        ax1.set_title(f'Distance Errors - {dataset_name}')
        ax1.set_xlabel('Prediction Number')
        ax1.set_ylabel('3D Distance Error (meters)')
        ax1.grid(True, alpha=0.3)
        
        # Plot cumulative error distribution
        ax2 = axes[idx, 1]
        has_cumulative_data = False
        
        # Plot Ben's cumulative distribution if available
        if ben_res and ben_res['distances']:
            ben_sorted = np.sort(ben_res['distances'])
            ben_cumulative = np.arange(1, len(ben_sorted) + 1) / len(ben_sorted)
            ax2.plot(ben_sorted, ben_cumulative, 'b-', label='Ben', linewidth=2)
            has_cumulative_data = True
        
        # Plot Yorgo's cumulative distribution if available
        if yorgo_res and yorgo_res['distances']:
            yorgo_sorted = np.sort(yorgo_res['distances'])
            yorgo_cumulative = np.arange(1, len(yorgo_sorted) + 1) / len(yorgo_sorted)
            ax2.plot(yorgo_sorted, yorgo_cumulative, 'r-', label='Yorgo', linewidth=2)
            has_cumulative_data = True
        
        if has_cumulative_data:
            ax2.legend()
        else:
            ax2.text(0.5, 0.5, 'No successful predictions\nto display', ha='center', va='center', transform=ax2.transAxes)
        
        ax2.set_title(f'Cumulative Error Distribution - {dataset_name}')
        ax2.set_xlabel('3D Distance Error (meters)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('flight_predictor_error_trends.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved as 'flight_predictor_error_trends.png'")
    
    plt.show()

def create_summary_table(all_results):
    """
    Create a summary table of results
    
    Args:
        all_results: Dictionary containing benchmark results
    
    Returns:
        pandas.DataFrame: Summary table
    """
    summary_data = []
    
    for dataset_name, results in all_results.items():
        for predictor_name in ['ben', 'yorgo']:
            if predictor_name in results and results[predictor_name] is not None:
                res = results[predictor_name]
                
                if res['successful_predictions'] > 0:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Predictor': predictor_name.capitalize(),
                        'Success Rate (%)': f"{res['success_rate']:.1f}",
                        'Mean Distance Error (m)': f"{res['mean_distance_error']:.2f}",
                        'Median Distance Error (m)': f"{res['median_distance_error']:.2f}",
                        'Std Distance Error (m)': f"{res['std_distance_error']:.2f}",
                        'Mean Lat Error (deg)': f"{res['mean_lat_error']:.6f}",
                        'Mean Lon Error (deg)': f"{res['mean_lon_error']:.6f}",
                        'Mean Alt Error (m)': f"{res['mean_alt_error']:.2f}",
                        'Successful Predictions': res['successful_predictions'],
                        'Failed Predictions': res['failed_predictions']
                    })
                else:
                    summary_data.append({
                        'Dataset': dataset_name,
                        'Predictor': predictor_name.capitalize(),
                        'Success Rate (%)': "0.0",
                        'Mean Distance Error (m)': "N/A",
                        'Median Distance Error (m)': "N/A",
                        'Std Distance Error (m)': "N/A",
                        'Mean Lat Error (deg)': "N/A",
                        'Mean Lon Error (deg)': "N/A",
                        'Mean Alt Error (m)': "N/A",
                        'Successful Predictions': 0,
                        'Failed Predictions': res['failed_predictions']
                    })
    
    return pd.DataFrame(summary_data)

def main():
    """Main function to run the benchmark"""
    print("Flight Path Predictor Benchmark")
    print("=" * 60)
    
    # Parse command line arguments
    predictors = None
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['ben', 'yorgo']:
            predictors = [arg]
            print(f"Running benchmark for {arg.capitalize()}'s predictor only")
        else:
            print(f"Warning: Unknown predictor '{sys.argv[1]}'. Valid options are 'ben' or 'yorgo'. Running all predictors.")
    
    # Check if plotting libraries are available
    if not MATPLOTLIB_AVAILABLE:
        print("Warning: matplotlib not available. Plots will be skipped.")
    elif not SEABORN_AVAILABLE:
        print("Note: seaborn not available. Using basic matplotlib styling.")
    
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
    results = run_benchmark(n_points, predictors=predictors)
    
    if results:
        print(f"\nBenchmark completed successfully!")
        if MATPLOTLIB_AVAILABLE:
            print(f"Visualizations have been generated and saved.")
        print(f"Results stored in memory for further analysis if needed.")
    else:
        print("Benchmark failed to complete.")

if __name__ == "__main__":
    main()