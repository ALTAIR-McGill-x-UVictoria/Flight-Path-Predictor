# format telemetry data for flight path prediction

import pandas as pd
import re
from datetime import datetime

# telemetry data location
folder = "telemetry_logs/"
# path = "flight_log_2025-07-03_22-10-39.txt"
path = "flight_log_2025-07-23_21-41-08.txt"

def parse_fc_telemetry(log_file_path):
    """
    Parse flight computer telemetry data from log files.
    
    Based on the C code format string:
    "%d,%d,%d,,%lu,%.6f,%.6f,%.2f,%.2f,%llu,,,%.2f,%.2f,%.2f,%d,%d,%d,%lu,%lu,,%lu,,,,,,.2f,%.2f,,,,%d,%d,%.2f,%.2f"
    
    Field mapping:
    0: ack (int)
    1: RSSI (int) 
    2: SNR (int)
    3: (empty - fc_unix_time_usec)
    4: fc_boot_time_ms (ulong)
    5: gpsLat2 (float, 6 decimal places)
    6: gpsLon2 (float, 6 decimal places) 
    7: gpsAlt2 (float, 2 decimal places)
    8: gpsSpeed2 (float, 2 decimal places)
    9: gpsTime2 (ulonglong)
    10: (empty - absPressure1)
    11: (empty - temperature1) 
    12: (empty - altitude1)
    13: absPressure2 (float, 2 decimal places)
    14: temperature2 (float, 2 decimal places)
    15: altitude1 (float, 2 decimal places)
    16: SDStatus (int, 1 or 0)
    17: actuatorStatus (int, 1 or 0)
    18: logging_active (int, 1 or 0)
    19: write_rate (ulong)
    20: space_left (ulong)
    21: (empty - pix_unix_time_usec)
    22: pix_boot_time_ms (ulong)
    23-28: (empty - vibration data)
    29: gpsBearing (float, 2 decimal places)
    30: gpsBearingMagnetic (float, 2 decimal places)
    31-35: (empty - other bearing data)
    36: photodiodeValue1 (int)
    37: photodiodeValue2 (int)
    38: FC_battery_voltage (float, 2 decimal places)
    39: LED_battery_voltage (float, 2 decimal places)
    """
    
    telemetry_data = []
    
    with open(log_file_path, 'r') as file:
        for line in file:
            # Look for lines that contain FC: packets
            if 'FC:' in line:
                # Extract timestamp from the log format [HH:MM:SS.mmm]
                timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]', line)
                if timestamp_match:
                    timestamp = timestamp_match.group(1)
                else:
                    timestamp = None
                
                # Extract the FC packet data (everything after 'FC:')
                fc_match = re.search(r'FC:(.+)', line)
                if fc_match:
                    fc_data = fc_match.group(1).strip()
                    
                    # Split by comma and handle empty fields
                    fields = fc_data.split(',')
                    
                    # Ensure we have enough fields (pad with empty strings if needed)
                    while len(fields) < 38:
                        fields.append('')
                    
                    try:
                        # Parse fields according to the C code format
                        telemetry_record = {
                            'timestamp': timestamp,
                            'ack': int(fields[0]) if fields[0] else None,
                            'rssi': int(fields[1]) if fields[1] else None,
                            'snr': int(fields[2]) if fields[2] else None,
                            'fc_unix_time_usec': None,  # Empty field
                            'fc_boot_time_ms': int(fields[4]) if fields[4] else None,
                            'gps_lat': float(fields[5]) if fields[5] else None,
                            'gps_lon': float(fields[6]) if fields[6] else None,
                            'gps_alt': float(fields[7]) if fields[7] else None,
                            'gps_speed': float(fields[8]) if fields[8] else None,
                            'gps_time': int(fields[9]) if fields[9] else None,
                            'abs_pressure1': None,  # Empty field
                            'temperature1': None,   # Empty field
                            'altitude1_empty': None,  # Empty field
                            'abs_pressure2': float(fields[13]) if fields[13] else None,
                            'temperature2': float(fields[14]) if fields[14] else None,
                            'altitude1': float(fields[15]) if fields[15] else None,
                            'sd_status': int(fields[16]) if fields[16] else None,
                            'actuator_status': int(fields[17]) if fields[17] else None,
                            'logging_active': int(fields[18]) if fields[18] else None,
                            'write_rate': int(fields[19]) if fields[19] else None,
                            'space_left': int(fields[20]) if fields[20] else None,
                            'pix_unix_time_usec': None,  # Empty field
                            'pix_boot_time_ms': int(fields[22]) if fields[22] else None,
                            'vibe_x': None,  # Empty field
                            'vibe_y': None,  # Empty field
                            'vibe_z': None,  # Empty field
                            'clip_x': None,  # Empty field
                            'clip_y': None,  # Empty field
                            'clip_z': None,  # Empty field
                            'gps_bearing': float(fields[29]) if fields[29] else None,
                            'gps_bearing_magnetic': float(fields[30]) if fields[30] else None,
                            'gps_bearing_true': None,  # Empty field
                            'gps_bearing_ground_speed': None,  # Empty field
                            'gps_bearing_ground_speed_magnetic': None,  # Empty field
                            'gps_bearing_ground_speed_true': None,  # Empty field
                            # Fixed: corrected field indices based on actual data structure (38 fields total)
                            'photodiode_value1': int(fields[34]) if len(fields) > 34 and fields[34] else None,
                            'photodiode_value2': int(fields[35]) if len(fields) > 35 and fields[35] else None,
                            'fc_battery_voltage': float(fields[36]) if len(fields) > 36 and fields[36] else None,
                            'led_battery_voltage': float(fields[37]) if len(fields) > 37 and fields[37] else None
                        }
                        
                        telemetry_data.append(telemetry_record)
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line.strip()}")
                        print(f"Error: {e}")
                        continue
    
    return pd.DataFrame(telemetry_data)

def load_and_parse_telemetry(file_path):
    """
    Load and parse telemetry data, returning a pandas DataFrame
    """
    print(f"Parsing telemetry data from: {file_path}")
    df = parse_fc_telemetry(file_path)
    print(f"Parsed {len(df)} telemetry records")
    
    # Display basic info about the data
    if not df.empty:
        print("\nData Summary:")
        print(f"Time range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
        print(f"GPS coordinates range:")
        print(f"  Latitude: {df['gps_lat'].min():.6f} to {df['gps_lat'].max():.6f}")
        print(f"  Longitude: {df['gps_lon'].min():.6f} to {df['gps_lon'].max():.6f}")
        print(f"  Altitude: {df['gps_alt'].min():.2f} to {df['gps_alt'].max():.2f} m")
        
    return df

# Example usage
if __name__ == "__main__":

    # Full path
    full_path = folder + path

    # Parse the telemetry data
    telemetry_df = load_and_parse_telemetry(full_path)

    # Display first few records
    print("\nFirst 5 telemetry records:")
    print(telemetry_df.head())
    
    # Save to CSV for further analysis
    output_path = path.replace('.txt', '_parsed.csv')
    telemetry_df.to_csv("parsed_data/" +output_path, index=False)
    print(f"\nParsed data saved to: {output_path}")

