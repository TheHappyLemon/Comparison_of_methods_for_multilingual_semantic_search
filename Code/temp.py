from datetime import timedelta
size = 10000

# time_str = input("Give me time string in format 'hh:mm:ss.ms' = ")
# hours, minutes, seconds = map(float, time_str.split(':'))
# time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds)
# seconds = time_delta.total_seconds()
# microsecs = time_delta.microseconds
# minutes = seconds / 60
# hours   = minutes / 60

# print(microsecs)

# print('timedelta = ', time_delta)
# print('microsecs per record =', microsecs / size)
# print('seconds per record   =', seconds / size)
# print('minutes per record   =', minutes / size)
# print('hous per record      =', hours / size)

# just for laser source en
time_delta_given = timedelta(days=1, hours=5, minutes=57, seconds=39, microseconds=559462)
minutes = time_delta_given.total_seconds() / 60
seconds = time_delta_given.total_seconds()
print('seconds per record = ', seconds / size)
print('minutes per record = ', minutes / size)