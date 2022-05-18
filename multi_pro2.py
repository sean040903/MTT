import time
import concurrent.futures
import numpy as np
from numba import jit
import multiprocessing

def do_something(i):
    a = np.empty(0)
    for j in range(i+1):
        a = np.append(a,j)
    return a


if __name__ == '__main__':

    start = time.perf_counter()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in range(1000):
            print(do_something(i))

    finish = time.perf_counter()

    print(f'{round(finish - start, 2)}초 만에 작업이 완료되었습니다.')