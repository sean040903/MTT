from multiprocess import Pool

p = Pool(4)

if __name__ == '__main__':
    print (p.map(lambda x: (lambda y:y**2)(x) + x, range(10)))
