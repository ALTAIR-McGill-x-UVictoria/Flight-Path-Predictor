import pandas as pd
import os
import numpy as np
from collections import deque
import time


class AdaptiveLearning:
    """
    Adaptive learning system that tracks prediction accuracy and adjusts parameters.
    """
    def __init__(self, max_history=100):
        self.prediction_history = deque(maxlen=max_history)
        self.accuracy_metrics = {
            'lat_errors': deque(maxlen=50),
            'lon_errors': deque(maxlen=50),
            'alt_errors': deque(maxlen=50),
            'method_weights': {
                'velocity': 0.4,
                'polynomial': 0.4,
                'pressure': 0.2
            },
            'phase_specific_weights': {
                'ascent': {'velocity': 0.5, 'polynomial': 0.3, 'pressure': 0.2},
                'float': {'velocity': 0.3, 'polynomial': 0.5, 'pressure': 0.2},
                'descent': {'velocity': 0.4, 'polynomial': 0.2, 'pressure': 0.4}
            }
        }
        self.smoothing_factors = {
            'lat': 0.3,
            'lon': 0.3, 
            'alt': 0.4
        }
        
    def update_prediction_accuracy(self, predicted, actual, flight_phase):
        """Update accuracy metrics based on prediction vs actual results."""
        lat_error = abs(predicted[0] - actual[0])
        lon_error = abs(predicted[1] - actual[1]) 
        alt_error = abs(predicted[2] - actual[2])
        
        self.accuracy_metrics['lat_errors'].append(lat_error)
        self.accuracy_metrics['lon_errors'].append(lon_error)
        self.accuracy_metrics['alt_errors'].append(alt_error)
        
        # Adapt method weights based on recent performance
        self._adapt_method_weights(flight_phase, lat_error, lon_error, alt_error)
        
    def _adapt_method_weights(self, flight_phase, lat_error, lon_error, alt_error):
        """Adapt method weights based on recent error patterns."""
        if len(self.accuracy_metrics['lat_errors']) < 10:
            return
            
        recent_errors = {
            'lat': np.mean(list(self.accuracy_metrics['lat_errors'])[-10:]),
            'lon': np.mean(list(self.accuracy_metrics['lon_errors'])[-10:]),
            'alt': np.mean(list(self.accuracy_metrics['alt_errors'])[-10:])
        }
        
        # Adjust smoothing factors based on error patterns
        for coord, error in recent_errors.items():
            if error < 0.00001:  # Very accurate predictions
                self.smoothing_factors[coord] = max(0.1, self.smoothing_factors[coord] - 0.05)
            elif error > 0.0001:  # Less accurate predictions  
                self.smoothing_factors[coord] = min(0.6, self.smoothing_factors[coord] + 0.05)
                
    def get_adaptive_weights(self, flight_phase):
        """Get current adaptive weights for prediction methods."""
        return self.accuracy_metrics['phase_specific_weights'].get(flight_phase, 
                                                                   self.accuracy_metrics['method_weights'])
    
    def get_smoothing_factor(self, coordinate):
        """Get adaptive smoothing factor for coordinate."""
        return self.smoothing_factors.get(coordinate, 0.3)


# Global adaptive learning instance
adaptive_learner = AdaptiveLearning()


def enhanced_data_validation(df):
    """
    Enhanced validation to detect and handle data quality issues.
    
    Args:
        df: DataFrame with GPS data
        
    Returns:
        tuple: (is_valid, cleaned_df, quality_score)
    """
    if len(df) < 2:
        return False, df, 0.0
        
    # Check for required columns
    required_cols = ['gps_lat', 'gps_lon', 'gps_alt']
    if not all(col in df.columns for col in required_cols):
        return False, df, 0.0
    
    quality_score = 1.0
    cleaned_df = df.copy()
    
    # Check for NaN values
    for col in required_cols:
        nan_ratio = df[col].isna().sum() / len(df)
        if nan_ratio > 0.5:  # More than 50% NaN
            return False, df, 0.0
        quality_score -= nan_ratio * 0.2
        
        # Forward fill small gaps
        if nan_ratio > 0:
            cleaned_df[col] = cleaned_df[col].fillna(method='ffill').fillna(method='bfill')
    
    # Check for unrealistic coordinate jumps
    for i, col in enumerate(['gps_lat', 'gps_lon', 'gps_alt']):
        coords = cleaned_df[col].values
        if len(coords) > 1:
            diffs = np.abs(np.diff(coords))
            
            # Define reasonable thresholds
            thresholds = [0.01, 0.01, 100]  # lat, lon (degrees), alt (meters)
            outliers = diffs > thresholds[i]
            
            if np.any(outliers):
                outlier_ratio = np.sum(outliers) / len(diffs)
                quality_score -= outlier_ratio * 0.3
                
                # Smooth out extreme outliers
                if outlier_ratio < 0.3:  # If not too many outliers
                    coords_smoothed = coords.copy()
                    outlier_indices = np.where(outliers)[0] + 1  # +1 because diff removes one element
                    
                    for idx in outlier_indices:
                        if 0 < idx < len(coords_smoothed) - 1:
                            # Replace with interpolated value
                            coords_smoothed[idx] = (coords_smoothed[idx-1] + coords_smoothed[idx+1]) / 2
                    
                    cleaned_df[col] = coords_smoothed
    
    # Check time consistency if available
    if 'gps_time' in cleaned_df.columns:
        times = cleaned_df['gps_time'].values
        if len(times) > 1:
            time_diffs = np.diff(times)
            if np.any(time_diffs <= 0):  # Non-monotonic time
                quality_score -= 0.2
            
            # Check for reasonable time intervals (1s to 60s typical)
            unreasonable_intervals = (time_diffs < 0.5) | (time_diffs > 120)
            if np.any(unreasonable_intervals):
                quality_score -= np.sum(unreasonable_intervals) / len(time_diffs) * 0.2
    
    is_valid = quality_score > 0.3  # Minimum quality threshold
    return is_valid, cleaned_df, max(0.0, quality_score)


def adaptive_smooth_gps_data(coords, coordinate_type='lat', flight_phase='float'):
    """
    Adaptive GPS smoothing that adjusts based on coordinate type and flight phase.
    
    Args:
        coords: Array of coordinate values
        coordinate_type: 'lat', 'lon', or 'alt'
        flight_phase: Current flight phase
    
    Returns:
        numpy.array: Adaptively smoothed coordinates
    """
    if len(coords) < 3:
        return coords
    
    # Get adaptive smoothing factor
    base_smoothing = adaptive_learner.get_smoothing_factor(coordinate_type)
    
    # Phase-specific adjustments
    phase_multipliers = {
        'ascent': {'lat': 0.8, 'lon': 0.8, 'alt': 1.2},  # More altitude smoothing
        'float': {'lat': 1.0, 'lon': 1.0, 'alt': 0.8},   # Less altitude smoothing
        'descent': {'lat': 0.9, 'lon': 0.9, 'alt': 1.1}  # Moderate altitude smoothing
    }
    
    phase_factor = phase_multipliers.get(flight_phase, {}).get(coordinate_type, 1.0)
    smoothing_factor = base_smoothing * phase_factor
    
    # Apply exponential smoothing
    smoothed = np.copy(coords).astype(float)
    for i in range(1, len(coords)):
        smoothed[i] = smoothing_factor * coords[i] + (1 - smoothing_factor) * smoothed[i-1]
    
    return smoothed


def advanced_validate_prediction_sanity(predicted_change, recent_changes, coordinate_type="coordinate", 
                                       flight_phase="float", data_quality=1.0):
    """
    Advanced validation with adaptive thresholds and quality-aware constraints.
    
    Args:
        predicted_change: The predicted change in coordinate
        recent_changes: Array of recent changes in the same coordinate
        coordinate_type: Type of coordinate (for error messages)
        flight_phase: Current flight phase
        data_quality: Quality score of input data (0-1)
    
    Returns:
        tuple: (is_valid, clamped_change, confidence_score)
    """
    if len(recent_changes) == 0:
        return True, predicted_change, 0.5
    
    # Calculate enhanced statistics
    recent_changes = np.array(recent_changes)
    median_change = np.median(np.abs(recent_changes))
    std_change = np.std(recent_changes)
    max_recent = np.max(np.abs(recent_changes))
    iqr = np.percentile(np.abs(recent_changes), 75) - np.percentile(np.abs(recent_changes), 25)
    
    # Adaptive thresholds based on flight phase and data quality
    phase_multipliers = {
        'ascent': {'lat': 2.0, 'lon': 2.0, 'alt': 3.0},
        'float': {'lat': 1.5, 'lon': 1.5, 'alt': 1.2},
        'descent': {'lat': 2.5, 'lon': 2.5, 'alt': 4.0}
    }
    
    coord_short = coordinate_type.split('_')[-1]  # Extract 'lat', 'lon', 'alt'
    phase_mult = phase_multipliers.get(flight_phase, {}).get(coord_short, 2.0)
    
    # Quality-adjusted threshold - lower quality data allows more variation
    quality_factor = 0.5 + 0.5 * data_quality  # Range: 0.5 to 1.0
    
    # Multiple validation approaches
    # 1. IQR-based threshold (robust to outliers)
    iqr_threshold = max(iqr * 2.5 * phase_mult, median_change * 2) * quality_factor
    
    # 2. Standard deviation threshold
    std_threshold = max(std_change * 2.5 * phase_mult, median_change) * quality_factor
    
    # 3. Historical maximum threshold
    max_threshold = max_recent * 2.0 * phase_mult * quality_factor
    
    # Use the most permissive threshold to avoid over-constraining
    max_allowed_change = max(iqr_threshold, std_threshold, max_threshold)
    
    predicted_magnitude = abs(predicted_change)
    
    # Calculate confidence based on how reasonable the prediction is
    confidence_score = 1.0
    if predicted_magnitude > max_allowed_change and max_allowed_change > 0:
        # Calculate how far outside reasonable bounds
        excess_ratio = predicted_magnitude / max_allowed_change
        confidence_score = max(0.1, 1.0 / excess_ratio)
        
        # Progressive clamping - less aggressive for small excesses
        if excess_ratio < 1.5:
            # Minor excess - light clamping
            scale_factor = 0.8
        elif excess_ratio < 3.0:
            # Moderate excess - moderate clamping
            scale_factor = max_allowed_change / predicted_magnitude
        else:
            # Major excess - aggressive clamping
            scale_factor = max_allowed_change / predicted_magnitude
        
        clamped_change = predicted_change * scale_factor
        
        print(f"Info: {coordinate_type} prediction adjusted from {predicted_change:.6f} to {clamped_change:.6f} "
              f"(excess: {excess_ratio:.1f}x, confidence: {confidence_score:.2f})")
        
        return excess_ratio < 3.0, clamped_change, confidence_score
    
    return True, predicted_change, confidence_score


def detect_flight_phase(df):
    """
    Enhanced flight phase detection with adaptive learning integration.
    
    Returns:
        str: 'ascent', 'float', or 'descent'
    """
    if len(df) < 3:
        return 'ascent'  # Default assumption
    
    alts = df['gps_alt'].values
    recent_alts = alts[-min(7, len(alts)):]  # Look at more points for better detection
    
    # Calculate altitude trend with multiple methods
    alt_trend = np.polyfit(range(len(recent_alts)), recent_alts, 1)[0]  # Linear slope
    alt_std = np.std(recent_alts)
    
    # Calculate velocity-based trend if GPS speed available
    velocity_trend = 0
    if 'gps_speed' in df.columns and not df['gps_speed'].isna().all():
        recent_speeds = df['gps_speed'].values[-min(5, len(df)):]
        valid_speeds = recent_speeds[~np.isnan(recent_speeds)]
        if len(valid_speeds) > 0:
            velocity_trend = np.mean(valid_speeds)
    
    # Enhanced thresholds with adaptive adjustment
    ASCENT_THRESHOLD = 0.8  # Slightly higher for more certainty
    DESCENT_THRESHOLD = -0.8
    FLOAT_STABILITY = 8.0   # Increased for better float detection
    
    # Multi-criteria decision with confidence weighting
    altitude_vote = 0
    if alt_trend > ASCENT_THRESHOLD:
        altitude_vote = 1  # ascent
    elif alt_trend < DESCENT_THRESHOLD:
        altitude_vote = -1  # descent
    
    stability_vote = 0
    if alt_std < FLOAT_STABILITY:
        stability_vote = 1  # suggests float
    
    # Combine evidence
    if altitude_vote == 1 and alt_std > FLOAT_STABILITY:
        return 'ascent'
    elif altitude_vote == -1 and alt_std > FLOAT_STABILITY:
        return 'descent'
    elif stability_vote == 1 and abs(alt_trend) < ASCENT_THRESHOLD:
        return 'float'
    else:
        # Ambiguous case - use additional data
        if velocity_trend > 15:  # High speed suggests ascent/descent
            return 'ascent' if alt_trend >= 0 else 'descent'
        elif velocity_trend < 5 and alt_std < FLOAT_STABILITY * 1.5:
            return 'float'
        else:
            # Final fallback based on altitude trend
            return 'ascent' if alt_trend >= 0 else 'descent'


def smooth_gps_data(coords, window_size=3):
    """
    Apply simple moving average smoothing to GPS coordinates to reduce noise.
    
    Args:
        coords: Array of coordinate values
        window_size: Size of smoothing window
    
    Returns:
        numpy.array: Smoothed coordinates
    """
    if len(coords) < window_size:
        return coords
    
    smoothed = np.copy(coords)
    half_window = window_size // 2
    
    for i in range(half_window, len(coords) - half_window):
        smoothed[i] = np.mean(coords[i-half_window:i+half_window+1])
    
    return smoothed
    """
    Apply simple moving average smoothing to GPS coordinates to reduce noise.
    
    Args:
        coords: Array of coordinate values
        window_size: Size of smoothing window
    
    Returns:
        numpy.array: Smoothed coordinates
    """
    if len(coords) < window_size:
        return coords
    
    smoothed = np.copy(coords)
    half_window = window_size // 2
    
    for i in range(half_window, len(coords) - half_window):
        smoothed[i] = np.mean(coords[i-half_window:i+half_window+1])
    
    return smoothed


def validate_prediction_sanity(predicted_change, recent_changes, coordinate_type="coordinate", 
                              flight_phase="float", data_quality=1.0):
    """
    Enhanced validation wrapper that calls the advanced validation function.
    """
    return advanced_validate_prediction_sanity(predicted_change, recent_changes, coordinate_type, 
                                              flight_phase, data_quality)


def fit_trajectory_polynomial(coords, timestamps=None, degree=2):
    """
    Fit a polynomial to the trajectory data to capture curved motion patterns.
    Enhanced with better validation and overfitting prevention.
    
    Args:
        coords: Array of coordinate values (lat, lon, or alt)
        timestamps: Array of time values, or None for uniform spacing
        degree: Degree of polynomial to fit (1=linear, 2=quadratic, 3=cubic)
    
    Returns:
        tuple: (coefficients, next_predicted_value, quality_score)
    """
    n_points = len(coords)
    if n_points < degree + 1:
        # Not enough points for polynomial fitting
        return None, None, 0.0
    
    # Limit polynomial degree to prevent overfitting
    max_safe_degree = min(degree, max(1, n_points // 2))  # Never exceed half the data points
    if max_safe_degree != degree:
        degree = max_safe_degree
    
    # Apply light smoothing to reduce GPS noise
    smoothed_coords = smooth_gps_data(coords, window_size=3)
    
    # Create x values (time indices or actual timestamps)
    if timestamps is not None and len(timestamps) == n_points:
        x_vals = np.array(timestamps)
        # Normalize to start from 0 for numerical stability
        x_vals = x_vals - x_vals[0]
        next_x = x_vals[-1] + (x_vals[-1] - x_vals[-2]) if len(x_vals) > 1 else x_vals[-1] + 1
    else:
        x_vals = np.arange(n_points)
        next_x = n_points
    
    try:
        # Fit polynomial
        coeffs = np.polyfit(x_vals, smoothed_coords, degree)
        poly = np.poly1d(coeffs)
        
        # Calculate R-squared to assess fit quality
        y_pred = poly(x_vals)
        ss_res = np.sum((smoothed_coords - y_pred) ** 2)
        ss_tot = np.sum((smoothed_coords - np.mean(smoothed_coords)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Predict next value
        next_val = poly(next_x)
        
        # Additional validation: check if prediction is reasonable
        recent_changes = np.diff(coords[-3:]) if len(coords) >= 3 else np.diff(coords)
        predicted_change = next_val - coords[-1]
        
        # If the polynomial prediction seems unrealistic, reduce confidence
        if len(recent_changes) > 0:
            typical_change = np.median(np.abs(recent_changes))
            if abs(predicted_change) > typical_change * 5:  # 5x typical change
                r_squared *= 0.5  # Reduce quality score
        
        return coeffs, next_val, r_squared
    except (np.linalg.LinAlgError, np.RankWarning, RuntimeWarning):
        # Polynomial fitting failed, fall back to lower degree
        if degree > 1:
            return fit_trajectory_polynomial(coords, timestamps, degree - 1)
        else:
            return None, None, 0.0


def get_wind_estimate(df):
    """
    Estimate wind velocity based on horizontal movement and GPS data.
    
    Returns:
        tuple: (wind_lat_component, wind_lon_component) in degrees/second
    """
    if len(df) < 2:
        return 0, 0
    
    lats = df['gps_lat'].values
    lons = df['gps_lon'].values
    
    # Calculate horizontal velocity (simplified as balloon movement)
    lat_vel = np.mean(np.diff(lats))
    lon_vel = np.mean(np.diff(lons))
    
    # If GPS speed and bearing are available, use them for better estimates
    if 'gps_speed' in df.columns and 'gps_bearing' in df.columns:
        speeds = df['gps_speed'].values
        bearings = df['gps_bearing'].values
        
        valid_data = ~(np.isnan(speeds) | np.isnan(bearings))
        if np.any(valid_data):
            avg_speed = np.mean(speeds[valid_data])  # m/s
            avg_bearing = np.mean(bearings[valid_data])  # degrees
            
            # Convert to lat/lon components (rough approximation)
            bearing_rad = np.radians(avg_bearing)
            # Convert m/s to degrees/s (very rough approximation)
            lat_component = avg_speed * np.cos(bearing_rad) * 9e-6  # Rough conversion
            lon_component = avg_speed * np.sin(bearing_rad) * 9e-6
            
            return lat_component, lon_component
    
    return lat_vel, lon_vel


# write code in this function to predict next point
def predict_next_point(df):
    """
    Predict the next GPS point based on flight phase, wind patterns, and sensor data.
    Enhanced with adaptive learning and improved validation.
    
    Args:
        df: DataFrame containing the last N GPS points with columns 'gps_lat', 'gps_lon', 'gps_alt'
    
    Returns:
        tuple: (predicted_lat, predicted_lon, predicted_alt)
    """
    try:
        # Enhanced data validation
        is_valid, df, data_quality = enhanced_data_validation(df)
        if not is_valid:
            print("Warning: Input data failed enhanced validation")
            return -1, -1, -1
        
        # Check if we have sufficient data
        if len(df) < 2:
            return -1, -1, -1
        
        # Extract GPS coordinates and apply enhanced smoothing
        raw_lats = df['gps_lat'].values
        raw_lons = df['gps_lon'].values
        raw_alts = df['gps_alt'].values
        
        # Check for valid data
        if np.any(np.isnan(raw_lats)) or np.any(np.isnan(raw_lons)) or np.any(np.isnan(raw_alts)):
            return -1, -1, -1
        
        # Detect current flight phase first for adaptive smoothing
        flight_phase = detect_flight_phase(df)
        
        # Apply adaptive smoothing based on flight phase
        lats = adaptive_smooth_gps_data(raw_lats, 'lat', flight_phase)
        lons = adaptive_smooth_gps_data(raw_lons, 'lon', flight_phase)
        alts = adaptive_smooth_gps_data(raw_alts, 'alt', flight_phase)
        
        n_points = len(df)
        
        # Get adaptive weights for this flight phase
        adaptive_weights = adaptive_learner.get_adaptive_weights(flight_phase)
        
        # Get wind estimate
        wind_lat, wind_lon = get_wind_estimate(df)
        
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
                            use_timestamps = True
                            processed_time_diffs = []
                            avg_positive_diff = np.mean(positive_diffs)
                            
                            for diff in time_diffs:
                                if np.isnan(diff) or diff <= 0:
                                    processed_time_diffs.append(avg_positive_diff)
                                else:
                                    processed_time_diffs.append(diff)
                            
                            time_diffs = np.array(processed_time_diffs)
                            print(f"Info: {flight_phase.capitalize()} phase - Using timestamps with {len(positive_diffs)}/{len(valid_diffs)} valid intervals, avg: {avg_positive_diff:.1f}s")
                        else:
                            print(f"Debug: All time differences are zero or negative")
                            print("Warning: No positive time differences found, falling back to unit intervals")
                    else:
                        print("Warning: No time differences calculated, falling back to unit intervals")
            except Exception as e:
                print(f"Warning: Could not process gps_time ({e}), using unit intervals")
        
        # Calculate velocity components for each interval with phase-aware weighting
        lat_velocities = []
        lon_velocities = []
        alt_velocities = []
        
        # Phase-specific weighting for historical data
        if flight_phase == 'ascent':
            # For ascent, prioritize recent vertical movement, less horizontal
            weights = np.exp(np.linspace(-1, 0, n_points-1))  # Exponential weighting favoring recent
        elif flight_phase == 'descent':
            # For descent, similar to ascent but consider parachute dynamics
            weights = np.exp(np.linspace(-0.5, 0, n_points-1))
        else:  # float phase
            # For float, horizontal movement is more consistent, use more uniform weighting
            weights = np.ones(n_points-1)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
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
        
        # Calculate weighted average velocities
        lat_velocities = np.array(lat_velocities)
        lon_velocities = np.array(lon_velocities)
        alt_velocities = np.array(alt_velocities)
        
        avg_lat_vel = np.average(lat_velocities, weights=weights)
        avg_lon_vel = np.average(lon_velocities, weights=weights)
        avg_alt_vel = np.average(alt_velocities, weights=weights)
        
        # Apply wind correction to horizontal components
        if flight_phase == 'float':
            # In float phase, movement is primarily wind-driven
            wind_factor = 0.8  # High influence of wind
            avg_lat_vel = avg_lat_vel * (1 - wind_factor) + wind_lat * wind_factor
            avg_lon_vel = avg_lon_vel * (1 - wind_factor) + wind_lon * wind_factor
        elif flight_phase == 'ascent':
            # In ascent, some wind influence but less
            wind_factor = 0.3
            avg_lat_vel = avg_lat_vel * (1 - wind_factor) + wind_lat * wind_factor
            avg_lon_vel = avg_lon_vel * (1 - wind_factor) + wind_lon * wind_factor
        # For descent, wind influence varies by parachute design - keep original for now
        
        # Use pressure data for altitude prediction if available
        pressure_alt_prediction = None
        if 'abs_pressure1' in df.columns and not df['abs_pressure1'].isna().all():
            try:
                pressures = df['abs_pressure1'].values[-min(3, len(df)):]
                pressures = pressures[~np.isnan(pressures)]
                if len(pressures) >= 2:
                    pressure_trend = np.polyfit(range(len(pressures)), pressures, 1)[0]
                    # Convert pressure change to altitude change (rough approximation)
                    # -12 Pa per meter at sea level, varies with altitude
                    current_alt = alts[-1]
                    pressure_factor = 12 * (1 + current_alt / 44330)  # Adjust for altitude
                    pressure_alt_change = -pressure_trend / pressure_factor
                    
                    if use_timestamps and len(time_diffs) > 0:
                        prediction_time_interval = np.mean(time_diffs[-min(3, len(time_diffs)):])
                        pressure_alt_prediction = pressure_alt_change * prediction_time_interval
                    else:
                        pressure_alt_prediction = pressure_alt_change
            except Exception as e:
                print(f"Warning: Could not use pressure data for altitude prediction: {e}")
        
        # Polynomial trajectory fitting for curved motion patterns
        poly_predictions = {'lat': None, 'lon': None, 'alt': None}
        poly_quality = {'lat': 0.0, 'lon': 0.0, 'alt': 0.0}
        
        # Be more conservative with polynomial degrees to prevent overfitting
        if flight_phase == 'float':
            # Float phase: start with linear, go to quadratic only with good data
            poly_degree = 2 if n_points >= 6 else 1
        elif flight_phase == 'ascent':
            # Ascent: usually linear, quadratic only with sufficient data
            poly_degree = 2 if n_points >= 5 else 1
        else:  # descent
            # Descent: conservative approach due to parachute complexity
            poly_degree = 1 if n_points < 5 else 2
        
        # Ensure we have minimum data for polynomial fitting
        min_points_needed = poly_degree + 2  # Need extra points for validation
        
        if n_points >= min_points_needed:
            # Prepare time data for polynomial fitting
            if use_timestamps and len(time_diffs) > 0:
                cumulative_times = np.cumsum([0] + list(time_diffs))
                timestamps_for_poly = cumulative_times
            else:
                timestamps_for_poly = None
            
            # Fit polynomials for each coordinate
            try:
                lat_coeffs, poly_lat, lat_quality = fit_trajectory_polynomial(lats, timestamps_for_poly, poly_degree)
                lon_coeffs, poly_lon, lon_quality = fit_trajectory_polynomial(lons, timestamps_for_poly, poly_degree)
                alt_coeffs, poly_alt, alt_quality = fit_trajectory_polynomial(alts, timestamps_for_poly, poly_degree)
                
                # Only use polynomial predictions if quality is reasonable
                MIN_QUALITY = 0.3  # Minimum R-squared to trust polynomial
                
                if poly_lat is not None and lat_quality > MIN_QUALITY:
                    poly_predictions['lat'] = poly_lat
                    poly_quality['lat'] = lat_quality
                if poly_lon is not None and lon_quality > MIN_QUALITY:
                    poly_predictions['lon'] = poly_lon
                    poly_quality['lon'] = lon_quality
                if poly_alt is not None and alt_quality > MIN_QUALITY:
                    poly_predictions['alt'] = poly_alt
                    poly_quality['alt'] = alt_quality
                    
                print(f"Info: Polynomial fitting (degree {poly_degree}) for {flight_phase} phase - Quality: lat={lat_quality:.2f}, lon={lon_quality:.2f}, alt={alt_quality:.2f}")
            except Exception as e:
                print(f"Warning: Polynomial fitting failed: {e}")
        else:
            print(f"Info: Insufficient data ({n_points}) for polynomial fitting, need {min_points_needed}")
        
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
            
            # Phase-specific weighting between recent direction and average velocity
            if flight_phase == 'ascent':
                # During ascent, vertical movement is more predictable, horizontal less so
                direction_weight = 0.8  # High weight on recent trend
                velocity_weight = 0.2
            elif flight_phase == 'descent':
                # During descent, movement can be more erratic initially, then stable
                direction_weight = 0.6
                velocity_weight = 0.4
            else:  # float phase
                # During float, movement is more consistent and wind-driven
                direction_weight = 0.5  # Balanced approach
                velocity_weight = 0.5
            
            # Calculate velocity-based predictions
            velocity_lat_change = (direction_weight * last_lat_dir) + (velocity_weight * avg_lat_change)
            velocity_lon_change = (direction_weight * last_lon_dir) + (velocity_weight * avg_lon_change)
            velocity_alt_change = (direction_weight * last_alt_dir) + (velocity_weight * avg_alt_change)
            
            # Combine polynomial and velocity-based predictions with quality-based weighting
            if poly_predictions['lat'] is not None:
                # Polynomial prediction available for latitude
                poly_lat_change = poly_predictions['lat'] - lats[-1]
                
                # Weight based on polynomial quality and data amount
                base_poly_weight = min(0.4, poly_quality['lat'])  # Max 40% weight
                if flight_phase == 'float' and n_points >= 6:
                    poly_weight = base_poly_weight * 1.2  # Slightly higher for float with good data
                else:
                    poly_weight = base_poly_weight * 0.8  # More conservative otherwise
                
                poly_weight = min(poly_weight, 0.5)  # Never exceed 50%
                
                # Validate the polynomial prediction with enhanced validation
                recent_lat_changes = np.diff(lats[-4:]) if len(lats) >= 4 else np.diff(lats)
                is_valid, validated_poly_change, confidence = validate_prediction_sanity(
                    poly_lat_change, recent_lat_changes, "latitude", flight_phase, data_quality)
                
                # Adjust polynomial weight based on validation confidence
                poly_weight *= confidence
                
                if is_valid and confidence > 0.5:
                    predicted_lat_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * velocity_lat_change)
                else:
                    # If polynomial prediction is questionable, reduce its weight significantly
                    poly_weight *= 0.3
                    predicted_lat_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * velocity_lat_change)
            else:
                predicted_lat_change = velocity_lat_change
            
            if poly_predictions['lon'] is not None:
                # Polynomial prediction available for longitude
                poly_lon_change = poly_predictions['lon'] - lons[-1]
                
                # Use same quality-based weighting as latitude
                base_poly_weight = min(0.4, poly_quality['lon'])
                if flight_phase == 'float' and n_points >= 6:
                    poly_weight = base_poly_weight * 1.2
                else:
                    poly_weight = base_poly_weight * 0.8
                
                poly_weight = min(poly_weight, 0.5)
                
                # Validate the polynomial prediction with enhanced validation
                recent_lon_changes = np.diff(lons[-4:]) if len(lons) >= 4 else np.diff(lons)
                is_valid, validated_poly_change, confidence = validate_prediction_sanity(
                    poly_lon_change, recent_lon_changes, "longitude", flight_phase, data_quality)
                
                # Adjust polynomial weight based on validation confidence
                poly_weight *= confidence
                
                if is_valid and confidence > 0.5:
                    predicted_lon_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * velocity_lon_change)
                else:
                    poly_weight *= 0.3
                    predicted_lon_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * velocity_lon_change)
            else:
                predicted_lon_change = velocity_lon_change
            
            # For altitude, combine polynomial, pressure, and velocity predictions
            if pressure_alt_prediction is not None and flight_phase in ['ascent', 'descent']:
                # Blend pressure-based and velocity-based predictions first
                pressure_weight = 0.7 if flight_phase == 'ascent' else 0.5
                combined_alt_change = (pressure_weight * pressure_alt_prediction) + ((1 - pressure_weight) * velocity_alt_change)
                
                # Then blend with polynomial if available (but be conservative)
                if poly_predictions['alt'] is not None:
                    poly_alt_change = poly_predictions['alt'] - alts[-1]
                    
                    # Validate altitude polynomial prediction with enhanced validation
                    recent_alt_changes = np.diff(alts[-4:]) if len(alts) >= 4 else np.diff(alts)
                    is_valid, validated_poly_change, confidence = validate_prediction_sanity(
                        poly_alt_change, recent_alt_changes, "altitude", flight_phase, data_quality)
                    
                    # Be very conservative with altitude polynomials
                    poly_weight = min(0.2, poly_quality['alt'] * 0.5 * confidence)  # Max 20% weight
                    
                    if is_valid and confidence > 0.4:
                        predicted_alt_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * combined_alt_change)
                    else:
                        # If altitude polynomial is questionable, ignore it
                        predicted_alt_change = combined_alt_change
                else:
                    predicted_alt_change = combined_alt_change
            elif poly_predictions['alt'] is not None:
                # Only polynomial and velocity predictions available
                poly_alt_change = poly_predictions['alt'] - alts[-1]
                
                # Validate altitude polynomial prediction with enhanced validation
                recent_alt_changes = np.diff(alts[-4:]) if len(alts) >= 4 else np.diff(alts)
                is_valid, validated_poly_change, confidence = validate_prediction_sanity(
                    poly_alt_change, recent_alt_changes, "altitude", flight_phase, data_quality)
                
                # Conservative weighting for altitude with confidence factor
                base_poly_weight = min(0.3, poly_quality['alt'] * confidence)
                if flight_phase == 'float' and n_points >= 6:
                    poly_weight = base_poly_weight
                else:
                    poly_weight = base_poly_weight * 0.7
                
                if is_valid and confidence > 0.4:
                    predicted_alt_change = (poly_weight * validated_poly_change) + ((1 - poly_weight) * velocity_alt_change)
                else:
                    # If altitude polynomial seems wrong, use mostly velocity
                    predicted_alt_change = (0.1 * validated_poly_change) + (0.9 * velocity_alt_change)
            else:
                predicted_alt_change = velocity_alt_change
        else:
            # Fallback to just average velocity, enhanced with polynomial if available
            if use_timestamps and len(time_diffs) > 0:
                prediction_time_interval = np.mean(time_diffs) if len(time_diffs) > 0 else 1.0
                velocity_lat_change = avg_lat_vel * prediction_time_interval
                velocity_lon_change = avg_lon_vel * prediction_time_interval
                velocity_alt_change = avg_alt_vel * prediction_time_interval
            else:
                velocity_lat_change = avg_lat_vel
                velocity_lon_change = avg_lon_vel
                velocity_alt_change = avg_alt_vel
            
            # Blend with polynomial predictions if available (fallback scenario)
            predicted_lat_change = velocity_lat_change
            predicted_lon_change = velocity_lon_change
            predicted_alt_change = velocity_alt_change
            
            if poly_predictions['lat'] is not None:
                poly_lat_change = poly_predictions['lat'] - lats[-1]
                predicted_lat_change = 0.3 * poly_lat_change + 0.7 * velocity_lat_change
            
            if poly_predictions['lon'] is not None:
                poly_lon_change = poly_predictions['lon'] - lons[-1]
                predicted_lon_change = 0.3 * poly_lon_change + 0.7 * velocity_lon_change
            
            if poly_predictions['alt'] is not None:
                poly_alt_change = poly_predictions['alt'] - alts[-1]
                predicted_alt_change = 0.3 * poly_alt_change + 0.7 * velocity_alt_change
        
        # Predict next point
        predicted_lat = lats[-1] + predicted_lat_change
        predicted_lon = lons[-1] + predicted_lon_change
        predicted_alt = alts[-1] + predicted_alt_change
        
        # Enhanced final sanity check on all predictions
        if n_points >= 3:
            # Get recent changes for final validation
            recent_lat_changes = np.diff(lats[-3:])
            recent_lon_changes = np.diff(lons[-3:])  
            recent_alt_changes = np.diff(alts[-3:])
            
            # Final validation pass with enhanced validation
            _, predicted_lat_change, lat_confidence = validate_prediction_sanity(
                predicted_lat_change, recent_lat_changes, "final_lat", flight_phase, data_quality)
            _, predicted_lon_change, lon_confidence = validate_prediction_sanity(
                predicted_lon_change, recent_lon_changes, "final_lon", flight_phase, data_quality)
            _, predicted_alt_change, alt_confidence = validate_prediction_sanity(
                predicted_alt_change, recent_alt_changes, "final_alt", flight_phase, data_quality)
            
            # Update predictions
            predicted_lat = lats[-1] + predicted_lat_change
            predicted_lon = lons[-1] + predicted_lon_change
            predicted_alt = alts[-1] + predicted_alt_change
            
            # Log confidence scores for monitoring
            avg_confidence = (lat_confidence + lon_confidence + alt_confidence) / 3
            if avg_confidence < 0.7:
                print(f"Info: Low prediction confidence ({avg_confidence:.2f}) for {flight_phase} phase")
        
        # Apply physics-based constraints for balloon flight
        current_alt = alts[-1]
        
        # Altitude constraints
        predicted_alt = max(0, predicted_alt)  # Cannot go below ground
        
        # Apply realistic altitude change limits based on flight phase
        if flight_phase == 'ascent':
            # Typical balloon ascent rate: 2-6 m/s, max ~10 m/s
            max_alt_change = 10 * (prediction_time_interval if use_timestamps and len(time_diffs) > 0 else 1)
            if predicted_alt_change > max_alt_change:
                predicted_alt = alts[-1] + max_alt_change
                print(f"Warning: Clamped excessive ascent rate from {predicted_alt_change:.1f} to {max_alt_change:.1f}")
        elif flight_phase == 'descent':
            # Typical parachute descent rate: 3-8 m/s, emergency descent can be higher
            max_descent_rate = -15 * (prediction_time_interval if use_timestamps and len(time_diffs) > 0 else 1)
            if predicted_alt_change < max_descent_rate:
                predicted_alt = alts[-1] + max_descent_rate
                print(f"Warning: Clamped excessive descent rate from {predicted_alt_change:.1f} to {max_descent_rate:.1f}")
        else:  # float phase
            # Float phase should have minimal altitude change
            max_float_change = 2 * (prediction_time_interval if use_timestamps and len(time_diffs) > 0 else 1)
            if abs(predicted_alt_change) > max_float_change:
                predicted_alt = alts[-1] + np.sign(predicted_alt_change) * max_float_change
        
        # Horizontal speed constraints (balloons don't move faster than strong winds)
        # Typical horizontal speeds: 0-30 m/s in strong jet streams
        if use_timestamps and len(time_diffs) > 0:
            time_int = prediction_time_interval
        else:
            time_int = 1.0
        
        # Convert lat/lon changes to approximate distances (very rough)
        lat_dist_approx = abs(predicted_lat_change) * 111000  # ~111 km per degree
        lon_dist_approx = abs(predicted_lon_change) * 111000 * np.cos(np.radians(predicted_lat))
        horizontal_speed = np.sqrt(lat_dist_approx**2 + lon_dist_approx**2) / time_int
        
        max_horizontal_speed = 50  # m/s (very strong winds)
        if horizontal_speed > max_horizontal_speed:
            scale_factor = max_horizontal_speed / horizontal_speed
            predicted_lat_change *= scale_factor
            predicted_lon_change *= scale_factor
            predicted_lat = lats[-1] + predicted_lat_change
            predicted_lon = lons[-1] + predicted_lon_change
            print(f"Warning: Clamped excessive horizontal speed from {horizontal_speed:.1f} to {max_horizontal_speed} m/s")
        
        return predicted_lat, predicted_lon, predicted_alt
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        return -1, -1, -1


def update_learning_feedback(predicted, actual, flight_phase):
    """
    Update the adaptive learning system with prediction accuracy feedback.
    Call this function when you have the actual next GPS point to compare with prediction.
    
    Args:
        predicted: tuple (predicted_lat, predicted_lon, predicted_alt)
        actual: tuple (actual_lat, actual_lon, actual_alt)  
        flight_phase: Current flight phase
    """
    try:
        adaptive_learner.update_prediction_accuracy(predicted, actual, flight_phase)
        
        # Calculate and log error metrics
        lat_error = abs(predicted[0] - actual[0]) * 111000  # Convert to meters (rough)
        lon_error = abs(predicted[1] - actual[1]) * 111000 * np.cos(np.radians(actual[0]))
        alt_error = abs(predicted[2] - actual[2])
        
        total_error = np.sqrt(lat_error**2 + lon_error**2 + alt_error**2)
        
        print(f"Adaptive Learning Update - {flight_phase} phase: "
              f"Total error: {total_error:.1f}m (lat: {lat_error:.1f}m, lon: {lon_error:.1f}m, alt: {alt_error:.1f}m)")
              
    except Exception as e:
        print(f"Error updating adaptive learning: {e}")


def get_prediction_confidence_metrics():
    """
    Get current confidence metrics from the adaptive learning system.
    
    Returns:
        dict: Current confidence and performance metrics
    """
    try:
        metrics = {
            'recent_errors': {
                'lat': np.mean(list(adaptive_learner.accuracy_metrics['lat_errors'])[-10:]) if len(adaptive_learner.accuracy_metrics['lat_errors']) > 0 else 0,
                'lon': np.mean(list(adaptive_learner.accuracy_metrics['lon_errors'])[-10:]) if len(adaptive_learner.accuracy_metrics['lon_errors']) > 0 else 0,
                'alt': np.mean(list(adaptive_learner.accuracy_metrics['alt_errors'])[-10:]) if len(adaptive_learner.accuracy_metrics['alt_errors']) > 0 else 0
            },
            'smoothing_factors': adaptive_learner.smoothing_factors.copy(),
            'method_weights': adaptive_learner.accuracy_metrics['method_weights'].copy(),
            'prediction_count': len(adaptive_learner.prediction_history)
        }
        return metrics
    except Exception as e:
        print(f"Error getting confidence metrics: {e}")
        return {}