import numpy as np




DCD = [0,3]
CAD = [3,4]
ADS = np.arange(5).reshape(1,-1)
print(ADS)
nds = 1
done1 = 1
ads = np.arange(5)
i = 0
while done1 == 1:
    j = 0
    re = 1
    while re * (nds - i) > 0:
        print(ADS)
        ADS1 = ADS[i]
        while re * (len(DCD) - j) > 0:
            k0 = DCD[j]
            k1 = CAD[j]
            ADS2 = np.copy(ADS1)
            a1 = ADS1[k1]
            a2 = ADS1[k0]
            ADS2[k0] = a1
            ADS2[k1] = a2
            i1 = 0
            l1 = len(ADS)
            while i1<l1:
                ads2 = ADS[i1]
                if np.array_equiv(ads2, ADS2):
                    j += 1
                    i1 = len(ADS)
                else:
                    i1 += 1
                    if i1 == len(ADS):
                        ADS = np.append(ADS, [ADS2],axis=0)
                        nds+=1
                        re = 0
        if re == 1:
            i += 1
        elif i< l1:
            i+=1
            j=0
    if re == 1:
        done1 = 0
print(ADS)

x, y= np.where(ADS == np.min(ADS))
print(x[0])
print(y[0])
list1 = ADS.tolist()

print(list1)
print(list1.index(min(list1)))