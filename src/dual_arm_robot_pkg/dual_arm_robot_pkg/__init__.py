from tqdm import tqdm
import time

for i in tqdm(range(1000), desc = "start Procesing", total = 100, leave = True, file = open('log.txt', 'w')):
    time.sleep(0.1)

