from __future__ import print_function, division, absolute_import

import math
import threading
from timeit import repeat

import numpy as np
from numba import jit, njit, prange, int64

nthreads = 4
size = 10**6

def func_np(a, b):
    """
    Control function using Numpy.
    """
    pil1 = np.empty(shape=(0), dtype=np.int64)
    for i in b:
        pil1 = np.append(pil1, np.sum(a <= i) - 1)
    return pil1

@jit('void(int64[:], int64[:], int64[:])', nopython=True, nogil=True)
def inner_func_nb(result, a, b):
    """
    Function under test.
    """
    for i in range(len(result)):
        result[i] = np.sum(a <= b[i]) - 1

def timefunc(correct, s, func, *args, **kwargs):
    """
    Benchmark *func* and print out its runtime.
    """
    print(s.ljust(20), end=" ")
    # Make sure the function is compiled before we start the benchmark
    res = func(*args, **kwargs)
    if correct is not None:
        assert np.allclose(res, correct), (res, correct)
    # time it
    print('{:>5.0f} ms'.format(min(repeat(lambda: func(*args, **kwargs),
                                          number=5, repeat=2)) * 1000))
    return res

def make_multithread(inner_func, numthreads):
    """
    Run the given function inside *numthreads* threads, splitting its
    arguments into equal-sized chunks.
    """
    def func_mt(*args):
        length = len(args[0])
        result = np.empty(length, dtype=np.int64)
        args = (result,) + args
        # Create argument tuples for each input chunk
        chunks = [[np.array_split(arg,numthreads)[i] for arg in args]
                  for i in range(numthreads)]
        # Spawn one thread per chunk
        threads = [threading.Thread(target=inner_func, args=chunk)
                   for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return result
    return func_mt

func_nb_mt = make_multithread(inner_func_nb, nthreads)

a = np.random.rand(size)
b = np.random.rand(size)

correct = timefunc(None, "numpy (1 thread)", func_np, a, b)
timefunc(correct, "numba (%d threads)" % nthreads, func_nb_mt, a, b)