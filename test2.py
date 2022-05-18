def NZBF(n):
    nzbf = []
    nbf = format(n, 'b')
    for i in range(len(nbf)):
        if nbf[i] == '1':
            nzbf.append(i)
    return nzbf

print(NZBF(9))
def NBF(n):
    nbf = []
    bf1=list(map(int,format(n,'b')))
    for i in range(len(format(n, 'b'))):
        nbf.append(bf1[- i - 1])
    return list(map(int, nbf))
print(NBF(NZBF(9)))