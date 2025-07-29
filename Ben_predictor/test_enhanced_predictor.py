#!/usr/bin/env python3
"""
Test script for the enhanced flight path predictor with adaptive learning.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add the current directory to the path to import the predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ben_predictor import predict_next_point, update_learning_feedback, get_prediction_confidence_metrics

def create_test_data():
    """Create synthetic test data for different flight phases."""
    
    # Ascent phase data
    ascent_data = {
        'gps_time': np.arange(1000, 1020, 2),  # 2-second intervals
        'gps_lat': np.array([45.5000, 45.5001, 45.5002, 45.5003, 45.5004, 
                            45.5005, 45.5006, 45.5007, 45.5008, 45.5009]),
        'gps_lon': np.array([-73.6000, -73.6001, -73.6002, -73.6003, -73.6004,
                            -73.6005, -73.6006, -73.6007, -73.6008, -73.6009]),
        'gps_alt': np.array([100, 150, 200, 250, 300, 350, 400, 450, 500, 550]),  # Ascending
        'abs_pressure1': np.array([1013, 1008, 1003, 998, 993, 988, 983, 978, 973, 968])
    }
    
    # Float phase data
    float_data = {
        'gps_time': np.arange(2000, 2020, 2),
        'gps_lat': np.array([45.6000, 45.6005, 45.6010, 45.6015, 45.6020,
                            45.6025, 45.6030, 45.6035, 45.6040, 45.6045]),
        'gps_lon': np.array([-73.5000, -73.5010, -73.5020, -73.5030, -73.5040,
                            -73.5050, -73.5060, -73.5070, -73.5080, -73.5090]),
        'gps_alt': np.array([15000, 15002, 14998, 15001, 14999, 15003, 14997, 15000, 15002, 14998]),  # Stable float
        'abs_pressure1': np.array([120, 121, 119, 120, 119, 121, 119, 120, 121, 119])
    }
    
    # Descent phase data
    descent_data = {
        'gps_time': np.arange(3000, 3020, 2),
        'gps_lat': np.array([45.7000, 45.7002, 45.7004, 45.7006, 45.7008,
                            45.7010, 45.7012, 45.7014, 45.7016, 45.7018]),
        'gps_lon': np.array([-73.4000, -73.4003, -73.4006, -73.4009, -73.4012,
                            -73.4015, -73.4018, -73.4021, -73.4024, -73.4027]),
        'gps_alt': np.array([10000, 9800, 9600, 9400, 9200, 9000, 8800, 8600, 8400, 8200]),  # Descending
        'abs_pressure1': np.array([300, 310, 320, 330, 340, 350, 360, 370, 380, 390])
    }
    
    return {
        'ascent': pd.DataFrame(ascent_data),
        'float': pd.DataFrame(float_data),
        'descent': pd.DataFrame(descent_data)
    }

def test_enhanced_predictor():
    """Test the enhanced predictor with different flight phases."""
    
    print("Testing Enhanced Flight Path Predictor with Adaptive Learning")
    print("=" * 60)
    
    test_data = create_test_data()
    
    for phase_name, df in test_data.items():
        print(f"\n--- Testing {phase_name.upper()} Phase ---")
        
        # Use first 8 points for prediction, last 2 for validation
        train_df = df.iloc[:8].copy()
        test_points = df.iloc[8:].copy()
        
        print(f"Training data: {len(train_df)} points")
        print(f"Test data: {len(test_points)} points")
        
        for i, (_, actual_row) in enumerate(test_points.iterrows()):
            print(f"\nPrediction #{i+1}:")
            
            # Make prediction
            pred_lat, pred_lon, pred_alt = predict_next_point(train_df)
            
            if pred_lat == -1:
                print("  Prediction failed!")
                continue
            
            # Get actual values
            actual_lat = actual_row['gps_lat']
            actual_lon = actual_row['gps_lon'] 
            actual_alt = actual_row['gps_alt']
            
            # Calculate errors (rough conversion to meters)
            lat_error = abs(pred_lat - actual_lat) * 111000
            lon_error = abs(pred_lon - actual_lon) * 111000 * np.cos(np.radians(actual_lat))
            alt_error = abs(pred_alt - actual_alt)
            total_error = np.sqrt(lat_error**2 + lon_error**2 + alt_error**2)
            
            print(f"  Predicted: ({pred_lat:.6f}, {pred_lon:.6f}, {pred_alt:.1f})")
            print(f"  Actual:    ({actual_lat:.6f}, {actual_lon:.6f}, {actual_alt:.1f})")
            print(f"  Errors: Lat={lat_error:.1f}m, Lon={lon_error:.1f}m, Alt={alt_error:.1f}m")
            print(f"  Total Error: {total_error:.1f}m")
            
            # Update adaptive learning
            update_learning_feedback(
                (pred_lat, pred_lon, pred_alt),
                (actual_lat, actual_lon, actual_alt),
                phase_name
            )
            
            # Add actual point to training data for next prediction
            train_df = pd.concat([train_df, actual_row.to_frame().T], ignore_index=True)
    
    # Show final confidence metrics
    print(f"\n--- Final Adaptive Learning Metrics ---")
    metrics = get_prediction_confidence_metrics()
    if metrics:
        print(f"Recent average errors:")
        print(f"  Latitude: {metrics['recent_errors']['lat']:.6f}")
        print(f"  Longitude: {metrics['recent_errors']['lon']:.6f}")
        print(f"  Altitude: {metrics['recent_errors']['alt']:.6f}")
        print(f"Current smoothing factors: {metrics['smoothing_factors']}")
        print(f"Predictions processed: {metrics['prediction_count']}")

if __name__ == "__main__":
    test_enhanced_predictor()
