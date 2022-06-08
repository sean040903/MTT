import time
import concurrent.futures
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool,Process,Queue,Lock, Value, Array
import os

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f1(name):
    info('function f')
    print('hello', name)

def f(name):
    print('hello', name)

def foo(q):
    q.put('hello')

def f2(q):
    q.put([42, None, 'hello'])

def f3(l, i):
    l.acquire()
    try:
        print('hello world', i)
    finally:
        l.release()

def f4(a1, a2):
    for i in range(len(a1)):
        a1[i] = a1[i]*a2[i]


if __name__ == '__main__':

    test = np.arange(10)
    for i in range(1,9):
        arr1 = Array('i', range(10))
        arr2 = Array('i', range(10,20))
        start = time.time()
        p = Process(target=f4, args=(arr1, arr2))
        p.start()
        p.join()
        end = time.time()
        print("수행시간: %f 초" % (end - start), "\n")

    start1 = time.time()
    np.arange(10)*np.arange(10,20)
    end1 = time.time()

    print("수행시간: %f 초" % (end1 - start1))