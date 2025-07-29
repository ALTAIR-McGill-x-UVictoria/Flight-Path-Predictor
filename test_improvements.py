#!/usr/bin/env python3
"""
Quick test script to evaluate the improved predictor
"""
import pandas as pd
import numpy as np
from Ben_predictor.ben_predictor import predict_next_point

def test_predictor():
    # Load some test data
    csv_file = 'parsed_data/flight_log_2025-07-03_22-10-39_parsed.csv'
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} data points")
    print(f"Columns: {list(df.columns)}")
    
    # Find approximate flight phases based on altitude
    altitudes = df['gps_alt'].values
    max_alt_idx = np.argmax(altitudes)
    max_altitude = altitudes[max_alt_idx]
    
    print(f"\nFlight Analysis:")
    print(f"Maximum altitude: {max_altitude:.1f}m at index {max_alt_idx}")
    print(f"Total flight duration: {len(df)} data points")
    
    # Test different flight phases
    test_scenarios = [
        # Early ascent
        {"name": "Early Ascent", "indices": [50, 100, 150]},
        # Mid ascent  
        {"name": "Mid Ascent", "indices": [300, 400, 500]},
        # Near apogee
        {"name": "Near Apogee", "indices": [max_alt_idx-10, max_alt_idx-5, max_alt_idx-2]},
        # Float/apogee phase
        {"name": "Float Phase", "indices": [max_alt_idx+2, max_alt_idx+5, max_alt_idx+10]},
        # Early descent (if available)
        {"name": "Early Descent", "indices": [max_alt_idx+50, max_alt_idx+100, max_alt_idx+150]}
    ]
    
    n_points = 5
    
    for scenario in test_scenarios:
        print(f"\n{'='*50}")
        print(f"TESTING: {scenario['name']}")
        print(f"{'='*50}")
        
        scenario_errors = []
        
        for start_idx in scenario['indices']:
            if start_idx < 0 or start_idx + n_points >= len(df):
                continue
                
            # Get test window
            test_data = df.iloc[start_idx:start_idx + n_points].copy()
            actual_next = df.iloc[start_idx + n_points]
            
            print(f"\n--- Testing at index {start_idx} ---")
            print(f"Altitude trend: {test_data['gps_alt'].values}")
            print(f"Actual next: lat={actual_next['gps_lat']:.6f}, lon={actual_next['gps_lon']:.6f}, alt={actual_next['gps_alt']:.1f}")
            
            # Make prediction
            try:
                pred_lat, pred_lon, pred_alt = predict_next_point(test_data)
                if pred_lat != -1:
                    lat_error = abs(pred_lat - actual_next['gps_lat'])
                    lon_error = abs(pred_lon - actual_next['gps_lon'])
                    alt_error = abs(pred_alt - actual_next['gps_alt'])
                    
                    print(f"Predicted: lat={pred_lat:.6f}, lon={pred_lon:.6f}, alt={pred_alt:.1f}")
                    print(f"Errors: lat={lat_error:.6f}°, lon={lon_error:.6f}°, alt={alt_error:.1f}m")
                    
                    # Calculate approximate distance error
                    lat_dist = lat_error * 111000  # rough conversion to meters
                    lon_dist = lon_error * 111000 * np.cos(np.radians(actual_next['gps_lat']))
                    dist_error = np.sqrt(lat_dist**2 + lon_dist**2)
                    print(f"Distance error: {dist_error:.1f} meters")
                    
                    scenario_errors.append(dist_error)
                else:
                    print("❌ Prediction failed")
            except Exception as e:
                print(f"❌ Error during prediction: {e}")
        
        # Summary for this scenario
        if scenario_errors:
            print(f"\n📊 {scenario['name']} Summary:")
            print(f"   Average error: {np.mean(scenario_errors):.1f} meters")
            print(f"   Min error: {np.min(scenario_errors):.1f} meters")
            print(f"   Max error: {np.max(scenario_errors):.1f} meters")
        else:
            print(f"\n❌ No successful predictions for {scenario['name']}")


def run_statistical_test():
    """Run a broader statistical test across the entire flight"""
    print(f"\n{'='*60}")
    print("STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    csv_file = 'parsed_data/flight_log_2025-07-03_22-10-39_parsed.csv'
    df = pd.read_csv(csv_file)
    
    n_points = 5
    test_every = 50  # Test every 50th point to get good coverage
    errors = []
    
    for i in range(0, len(df) - n_points, test_every):
        test_data = df.iloc[i:i + n_points].copy()
        actual_next = df.iloc[i + n_points]
        
        try:
            pred_lat, pred_lon, pred_alt = predict_next_point(test_data)
            if pred_lat != -1:
                lat_error = abs(pred_lat - actual_next['gps_lat'])
                lon_error = abs(pred_lon - actual_next['gps_lon'])
                
                lat_dist = lat_error * 111000
                lon_dist = lon_error * 111000 * np.cos(np.radians(actual_next['gps_lat']))
                dist_error = np.sqrt(lat_dist**2 + lon_dist**2)
                errors.append(dist_error)
        except:
            continue
    
    if errors:
        print(f"Tested {len(errors)} predictions across the flight:")
        print(f"📊 Overall Statistics:")
        print(f"   Mean error: {np.mean(errors):.1f} meters")
        print(f"   Median error: {np.median(errors):.1f} meters") 
        print(f"   Std deviation: {np.std(errors):.1f} meters")
        print(f"   Min error: {np.min(errors):.1f} meters")
        print(f"   Max error: {np.max(errors):.1f} meters")
        print(f"   95th percentile: {np.percentile(errors, 95):.1f} meters")
        
        # Error distribution
        under_10m = np.sum(np.array(errors) < 10)
        under_50m = np.sum(np.array(errors) < 50) 
        under_100m = np.sum(np.array(errors) < 100)
        
        print(f"\n📈 Error Distribution:")
        print(f"   < 10m: {under_10m}/{len(errors)} ({under_10m/len(errors)*100:.1f}%)")
        print(f"   < 50m: {under_50m}/{len(errors)} ({under_50m/len(errors)*100:.1f}%)")
        print(f"   < 100m: {under_100m}/{len(errors)} ({under_100m/len(errors)*100:.1f}%)")


if __name__ == "__main__":
    test_predictor()
    run_statistical_test()
