from reprint import output
import time
import random

def inter2():
    lenth = random.randint(1,10)
    with output(initial_len=lenth, interval=0) as output_lines:
        while True:
            inlen = random.randint(1,lenth)
            for i in range(inlen):
                output_lines[i] = "{}...".format(random.randint(1,10))
            for i in range(inlen,lenth):
                output_lines[i] = '...'
            time.sleep(0.01)
        time.sleep(0.1)
                
    
inter2()


