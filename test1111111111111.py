import numpy as np
import time

oap  = np.random.rand(1000)

%timeit np.argsort(oap)

%timeit np.sort(oap)
