import multiprocessing
import time
start_time = time.time()
def count(name):
    for i in range(50000):
        print(name,":",i)
numlist = ['p1','p2','p3','p4']
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=4)
    pool.map(count,numlist)
    pool.close()
    pool.join()

print("--- %s seconds ---" % (time.time()-start_time))