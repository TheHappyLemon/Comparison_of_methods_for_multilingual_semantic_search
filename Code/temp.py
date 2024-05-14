from datetime import timedelta



from datetime import timedelta

# Example timedelta string
time_str = "1 day, 5:57:39.559462"

# Parse days, hours, minutes, seconds
parts = time_str.split(', ')
days = int(parts[0].split(' ')[0])
time_parts = parts[1].split(':')
hours = int(time_parts[0])
minutes = int(time_parts[1])
seconds = float(time_parts[2])

# Create timedelta object
time_delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)

# Calculate total minutes
total_minutes = time_delta.total_seconds() / 60

print(total_minutes)
