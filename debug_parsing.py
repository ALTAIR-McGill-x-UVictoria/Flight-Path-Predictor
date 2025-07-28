import re

# Test with actual line from the log
test_line = "[22:10:41.580] FC:0,-26,12,,337644,45.863163,-73.592896,54.87,1.35,1751497180,,,,100679.30,34.65,53.90,1,0,0,0,0,,11742,,,,,,,0.00,274.00,,,,24,17,16.28,12.60"

print("Debugging FC packet parsing:")
print(f"Test line: {test_line}")

# Check if 'FC:' is in the line
print(f"'FC:' in line: {'FC:' in test_line}")

# Extract timestamp
timestamp_match = re.search(r'\[(\d{2}:\d{2}:\d{2}\.\d{3})\]', test_line)
if timestamp_match:
    timestamp = timestamp_match.group(1)
    print(f"Timestamp: {timestamp}")
else:
    print("No timestamp found")

# Extract FC data
fc_match = re.search(r'FC:(.+)', test_line)
if fc_match:
    fc_data = fc_match.group(1).strip()
    print(f"FC data: {fc_data}")
    
    # Split by comma
    fields = fc_data.split(',')
    print(f"Number of fields: {len(fields)}")
    
    # Show first 10 fields
    for i in range(min(10, len(fields))):
        print(f"Field {i}: '{fields[i]}'")
        
    # Show last few fields
    print("...")
    for i in range(max(0, len(fields)-5), len(fields)):
        print(f"Field {i}: '{fields[i]}'")
        
else:
    print("No FC data found")

print("\nTesting file read:")
file_path = "telemetry_logs/flight_log_2025-07-03_22-10-39.txt"
try:
    with open(file_path, 'r') as file:
        lines = file.readlines()[:5]  # First 5 lines
        print(f"File has {len(lines)} lines (showing first 5):")
        for i, line in enumerate(lines):
            print(f"Line {i}: {line.strip()}")
            if 'FC:' in line:
                print(f"  -> Found FC line!")
except FileNotFoundError:
    print(f"File not found: {file_path}")
except Exception as e:
    print(f"Error reading file: {e}")
