from datetime import timedelta

size = 10000

time_str = input("Give me time string in format 'hh:mm:ss.ms' = ")
hours, minutes, seconds = map(float, time_str.split(':'))
time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
minutes = time_delta.total_seconds() / 60
seconds = time_delta.total_seconds()

print('timedelta = ', time_delta)
print('minutes per record =', minutes / size)
print('seconds per record =', seconds / size)