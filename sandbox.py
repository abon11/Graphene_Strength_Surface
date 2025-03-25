import time
from datetime import timedelta

i = 1
start_time = time.perf_counter()
print("start", start_time)
count = 0

while i < 10:
    if count % 10000000 == 0:
        time1000 = time.perf_counter()
        print("now", time1000)
        print("elapsed", timedelta(seconds=time1000 - start_time))

    count += 1
