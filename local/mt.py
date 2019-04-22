def process():
    end = time.time()
    for i in range(10000000):
        i
    tt.append(time.time() - end)

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures as futures
import time
import multiprocessing
tt = []
futurelist = []
end = time.time()
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()*2) as executor:
    for eps in range(multiprocessing.cpu_count()*2):
        futurelist.append(executor.submit(process))
print(time.time()-end)
print(sum(tt)/len(tt))
tt = []
print('END')
end = time.time()
for eps in range(multiprocessing.cpu_count()*2):
    process()
print(time.time()-end)
print(sum(tt)/len(tt))
