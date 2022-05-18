import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# input 2-dimension lists#
print("input S")
S = np.array(map(int, input().split()))
s = len(S)
print("input C")
C = np.array(map(int, input().split()))
c = len(C)
print("input SC")
SC = [np.array(map(int, input().split())) for _ in range(s)]
print("input FCS")
FCS = np.array(map(int, input().split()))
print("input LCS")
LCS = np.array(map(int, input().split()))
print("input KC")
KC = [np.array(map(int, input().split())) for _ in range(c)]
Prime = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107,
         109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
         233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359,
         367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491,
         499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
         643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787,
         797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
         947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
         1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213,
         1217, 1223, 1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
         1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451, 1453, 1459, 1471, 1481,
         1483, 1487, 1489, 1493, 1499, 1511, 1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
         1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
         1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
         1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987, 1993, 1997, 1999, 2003, 2011, 2017,
         2027, 2029, 2039, 2053, 2063, 2069, 2081, 2083, 2087, 2089, 2099, 2111, 2113, 2129, 2131, 2137, 2141, 2143,
         2153, 2161, 2179, 2203, 2207, 2213, 2221, 2237, 2239, 2243, 2251, 2267, 2269, 2273, 2281, 2287, 2293, 2297,
         2309, 2311, 2333, 2339, 2341, 2347, 2351, 2357, 2371, 2377, 2381, 2383, 2389, 2393, 2399, 2411, 2417, 2423,
         2437, 2441, 2447, 2459, 2467, 2473, 2477, 2503, 2521, 2531, 2539, 2543, 2549, 2551, 2557, 2579, 2591, 2593,
         2609, 2617, 2621, 2633, 2647, 2657, 2659, 2663, 2671, 2677, 2683, 2687, 2689, 2693, 2699, 2707, 2711, 2713,
         2719, 2729, 2731, 2741, 2749, 2753, 2767, 2777, 2789, 2791, 2797, 2801, 2803, 2819, 2833, 2837, 2843, 2851,
         2857, 2861, 2879, 2887, 2897, 2903, 2909, 2917, 2927, 2939, 2953, 2957, 2963, 2969, 2971, 2999, 3001, 3011,
         3019, 3023, 3037, 3041, 3049, 3061, 3067, 3079, 3083, 3089, 3109, 3119, 3121, 3137, 3163, 3167, 3169, 3181,
         3187, 3191, 3203, 3209, 3217, 3221, 3229, 3251, 3253, 3257, 3259, 3271, 3299, 3301, 3307, 3313, 3319, 3323,
         3329, 3331, 3343, 3347, 3359, 3361, 3371, 3373, 3389, 3391, 3407, 3413, 3433, 3449, 3457, 3461, 3463, 3467,
         3469, 3491, 3499, 3511, 3517, 3527, 3529, 3533, 3539, 3541, 3547, 3557, 3559, 3571]
print("input MCpTpS")
MCpTpS = np.array(map(int, input().split()))
print("input DI")
DI = [np.array(map(int, input().split())) for _ in range(c)]

# SC sorting#
zCC = np.zeros((c, c))
cSC = SC


def CCS(n):
    return list(map(str, np.arange(0, n)))


TCS = [0] * s
for s1 in range(s):
    TCS[s1] = sum(SC[s1])

dfSC = pd.DataFrame(SC, columns=CCS(c))
dfSC['TCS'] = TCS
dfSC['S'] = np.arange(0, s)
dfSC.sort_values('TCS', ascending=True)
StS = np.zeros(s)
for i in range(s):
    StS[i] = dfSC.ix[i]['S']
dfSC.drop('TCS', axis=1, inplace=True)
dfSC.drop('S', axis=1, inplace=True)
dfSC = dfSC.rest_index(drop=True)

for s1 in range(s):
    for c1 in range(c):
        l1 = list(map(str, [c1]))
        cSC[s1][c1] = dfSC.ix[s1][l1[0]]

def TSC(mat):
    TSC = np.zeros(c)
    for s1 in range(s):
        TSC += [mat[s1]]
    return TSC

TSCs = []

# 반 분류#
Q = T = M = D = P = []
QU = TR = PA = []
m = d = q = t = p = 0
for c1 in range(c):
    if KC[c1][0] == 3:
        T.append(c1)
        t += 1
    if KC[c1][0] == 2:
        D.append(c1)
        d += 1
    if KC[c1][0] == 11:
        P.append(c1)
        p += 1
    if KC[c1][0] == 1:
        M.append(c1)
        m += 1
    else:
        Q.append(c1)
        q += 1


# 행렬-원소선택 함수들#
def PIL(mat, n):
    for i in range(len(mat)):
        if mat[i] >= n:
            return i


def NED(n, mat):
    for i in range(len(mat)):
        if n % mat[i] == 0:
            return 0
    return 1


# 소수,진법변환관련 함수들#
def Pm(n1, n2):
    multiply = 0
    while n1 % (Prime[n2] ** multiply) == 0:
        multiply = +1
        return multiply - 1


def NBF(n):
    nbf = []
    for i in range(len(format(n, 'b'))):
        nbf.append(format(n, 'b')[len(format(n, 'b')) - i - 1])
    return list(map(int, nbf))


def NZBF(n):
    nzbf = []
    nbf = format(n, 'b')
    for i in range(len(nbf)):
        if nbf[len(nbf) - i - 1] == '1':
            nzbf.append(i)
    return nzbf


def BFN(mat):
    n = 0
    for i in range(len(mat)):
        n += 2 ** mat[i]
    return n


# 행렬에서 연속 몇개 선택 함수들#
def MIO(mat, i, n):
    mio = 1
    for j in range(i, i + n):
        mio *= mat[j]
    return mio


def SelIO(mat, i, n):
    selio = []
    for j in range(i, i + n):
        selio.append(mat[j])
    return selio


# 행렬-행렬관계 함수#
def CIO(mat1, mat2):
    for i in range(len(mat1)):
        if mat1[i] > mat2[i]:
            return 1
        elif mat2[i] > mat1[i]:
            return 2
    return 0


# 인접한 교시거리관련 함수들#
def DDRP(mat, SCT):
    dis = []
    for i in range(2):
        for j in range(2):
            di = 0
            t1 = mat[0][i]
            t2 = mat[1][j]
            for s1 in range(s):
                c1 = SCT[s1][t1]
                c2 = SCT[s1][t2]
                di += DI[c1][c2]
            dis.append(di)
    return [min(dis), dis.index(min(dis))]


def DMRP(mat, SCT):
    dis = []
    for i in range(2):
        di = 0
        t1 = mat[0][i]
        t2 = mat[1][0]
        for s1 in range(s):
            c1 = SCT[s1][t1]
            c2 = SCT[s1][t2]
            di += DI[c1][c2]
        dis.append(di)
    return [min(dis), dis.index(min(dis))]


# 교시교환 관련 함수들#
DRP = [[0, 1, 2, 3, 4], [4, 1, 2, 3, 0]]
Mon = [0, 2, 4]
Tu = [6, 8, 10]
We = [13, 15]
Th = [17, 19, 21]
Fr = [24, 26, 28]
FTD = [0, 6, 13, 17, 24, 30]
NDS = [0, 4, 6, 10, 13, 17, 21, 24, 28]


def date(x):
    return PIL(FTD, x)


def sixTTRP(mat, SCT):
    L = Lij = []
    for k in range(3):
        a = mat[k]
        b = mat[k + 1]
        ddrp = DDRP([[a, a + 1], [b, b + 1]], SCT)
        L.append(ddrp[0])
        Lij.append(ddrp[1])
    Lt = L.index(min(L))
    lij = Lij[Lt]
    [j, i] = NBF(lij)
    a = mat[Lt]
    b = mat[Lt + 1]
    c = mat[Lt - 1]
    return [a + 1 - i, a + i, b + j, b + 1 - j, c, c + 1]


def sevenTTRP(mat, SCT):
    L = Lij1 = Lij2 = []
    f = mat[2] + 2
    for k in range(3):
        a = mat[k]
        b = mat[k + 1]
        c = mat[k - 1]
        ddrp = DDRP([[a, a + 1], [f, f]], SCT)
        dmrp = DMRP([[b, b + 1], [c, c + 1]], SCT)
        L.append(ddrp[0] + dmrp[0])
        Lij1.append(dmrp[1])
        Lij2.append(ddrp[1])
    Lt = L.index(min(L))
    lij1 = Lij1[Lt]
    lij2 = Lij2[Lt]
    i1 = lij1 % 2
    [j2, i2] = NBF(lij2)
    a = mat[Lt]
    b = mat[Lt + 1]
    c = mat[Lt - 1]
    return [b + 1 - i2, b + i2, c + j2, c + 1 - j2, a + 1 - i1, a + i1, f]


def fourTTRP(mat, SCT):
    a = mat[0]
    b = mat[0]
    ddrp = DDRP([[a, a + 1], [b, b + 1]], SCT)
    Lij = ddrp[1]
    [j, i] = NBF(Lij)
    return [a + 1 - i, a + i, b + j, b + 1 - j]


def thirtyTTRP(mat1, mat2, mat3, mat4, mat5, SCT):
    L1 = sixTTRP(mat1, SCT)
    L2 = sevenTTRP(mat2, SCT)
    L3 = fourTTRP(mat3, SCT)
    L4 = sevenTTRP(mat4, SCT)
    L5 = sixTTRP(mat5, SCT)
    return L1 + L2 + L3 + L4 + L5


# ETO,DETO관련 함수들#
ETO = [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 24, 26, 28]
DETO = [1, 3, 5, 7, 9, 11, 14, 16, 18, 20, 22, 25, 27, 29]


def ETOs(mat):
    etos = []
    for i in range(len(mat)):
        etos.append(ETO[mat[i]])
    return etos


def DETOs(mat):
    detos = []
    for i in range(len(mat)):
        detos.append(DETO[mat[i]])
    return detos


def Even(TType):
    list1 = ETOs(np.arange(0, TType[0] + 1))
    list2 = ETOs(np.arange(3, TType[1] + 1))
    list3 = ETOs(np.arange(6, TType[2] + 1))
    list4 = ETOs(np.arange(8, TType[3] + 1)) * TType[5]
    list5 = ETOs(np.arange(11, TType[4] + 1)) * TType[6]
    return list(set(list1) | set(list2) | set(list3) | set(list4) | set(list5))


def Odd(TType):
    list1 = ETOs(np.arange(0, TType[0] + 1))
    list2 = ETOs(np.arange(3, TType[1] + 1))
    list3 = ETOs(np.arange(6, TType[2] + 1))
    list4 = ETOs(np.arange(8, TType[3] + 1)) * TType[5]
    list5 = ETOs(np.arange(11, TType[4] + 1)) * TType[6]
    list6 = DETOs(NZBF(TType[8]))
    list7 = [23 * TType[7], 12]
    return list(set(list1) | set(list2) | set(list3) | set(list4) | set(list5) | set(list6) | set(list7))


# D,M코드에서 TType 변경하는 함수#
def CTType(a, b, TType):
    if TType[0] < 3:
        if b == ETO[TType[0]]:
            TType[0] += 1
    if TType[1] < 6:
        if b == ETO[TType[1]]:
            TType[1] += 1
    if TType[2] < 8:
        if b == ETO[TType[2]]:
            TType[2] += 1
    if TType[3] < 11:
        if b == ETO[TType[3]]:
            TType[3] += 1
    if TType[4] < 14:
        if b == ETO[TType[4]]:
            TType[4] += 1
    dat = date(b)
    if dat < 2:
        TType[6 - dat] = 1
    if (b - 23) * (b - 12) == 0:
        TType[5] = TType[7] = 1
    if a == 1:
        if not list({b} - set(ETO)):
            TType[8] = BFN(list(set(NZBF(TType[8])) | {ETO.index(b)}))
        if not list({b} - set(DETO)):
            TType[8] = BFN(list(set(NZBF(TType[8])) | {DETO.index(b)}))
    return TType


# DO,MO,QU,TR,PA 관련#
def DO(TType):
    return Even(TType)


def do(TType):
    return len(DO(TType))


def MO(TType):
    return Odd(TType)


def mo(TType):
    return len(MO(TType))


def TR(TType):
    TR1 = []
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                TR1.append([L1[t1], L2[t2]])
    return TR1


def TTType(TType):
    TTType1 = []
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                TTType1.append(CTType(1, L2[t2], ttype))
    return TTType1


def tr(TType):
    tr1 = 0
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                tr1 += 1
    return tr1


def QU(TType):
    QU1 = []
    L1 = Even(TType)
    don1 = do(TType)
    for t1 in range(don1):
        ttype = CTType(0, L1[t1], TType)
        L2 = Even(ttype)
        don2 = do(ttype)
        for t2 in range(don2):
            if date(L2[t2]) > date(L1[t1]):
                QU1.append([L1[t1], L2[t2]])
    return QU1


def QTType(TType):
    QTType1 = []
    L1 = Even(TType)
    don1 = do(TType)
    for t1 in range(don1):
        ttype = CTType(0, L1[t1], TType)
        L2 = Even(ttype)
        don2 = do(ttype)
        for t2 in range(don2):
            if date(L2[t2]) > date(L1[t1]):
                qTType = CTType(0, L2[t2], ttype)
                if TType[5] == 0:
                    if date(L1[t1]) == 2:
                        if L1[t1] + 11 == L2[t2]:
                            qTType[5] = 0
                if TType[6] == 0:
                    if date(L1[t1]) == 1:
                        if L1[t1] + 24 == L2[t2]:
                            qTType[6] = 0
                QTType1.append(qTType)
    return QTType1


def qu(TType):
    qu1 = 0
    L1 = Even(TType)
    don1 = do(TType)
    for t1 in range(don1):
        ttype = CTType(0, L1[t1], TType)
        L2 = Even(ttype)
        don2 = do(ttype)
        for t2 in range(don2):
            if date(L2[t2]) > date(L1[t1]):
                qu1 += 1
    return qu1


def PA(TType):
    PA1 = []
    L1 = Odd(TType)
    mon1 = mo(TType)
    for t1 in range(mon1):
        ttype = CTType(1, L1[t1], TType)
        L2 = Odd(ttype)
        mon2 = mo(ttype)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                PA1.append([L1[t1], L2[t2]])
    return PA1


def PTType(TType):
    PTType1 = []
    L1 = Odd(TType)
    mon1 = mo(TType)
    for t1 in range(mon1):
        ttype = CTType(1, L1[t1], TType)
        L2 = Odd(ttype)
        mon2 = mo(ttype)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                pTType = CTType(1, L2[t2], ttype)
                if TType[5] == 0:
                    if date(L1[t1]) == 2:
                        if L1[t1] + 11 == L2[t2]:
                            pTType[5] = 0
                if TType[6] == 0:
                    if date(L1[t1]) == 1:
                        if L1[t1] + 24 == L2[t2]:
                            pTType[6] = 0
                PTType1.append(pTType)
    return PTType1


def pa(TType):
    pa1 = 0
    L1 = Odd(TType)
    mon1 = mo(TType)
    for t1 in range(mon1):
        ttype = CTType(1, L1[t1], TType)
        L2 = Odd(ttype)
        mon2 = mo(ttype)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                pa1 += 1
    return pa1


# 그래프 색깔#
SPC = np.zeros(12)
GT = ['Mathematics', 'Physics', 'Chemistry', 'Computer Science', 'Life Science', 'Earth Science', 'Convergence Science',
      'Science Experiment', 'Creative Convergence Special Lecture', 'Language', 'Society', 'Music-Art-Sport']
Math = Phy = Che = CoSc = LiSc = EaSc = ConS = ScEx = CCSL = Lan = Soc = MAS = []


def CSP(n):
    if n == 0:
        return Math
    elif n == 1:
        return Phy
    elif n == 2:
        return Che
    elif n == 3:
        return CoSc
    elif n == 4:
        return LiSc
    elif n == 5:
        return EaSc
    elif n == 6:
        return ConS
    elif n == 7:
        return ScEx
    elif n == 8:
        return CCSL
    elif n == 9:
        return Lan
    elif n == 10:
        return Soc
    else:
        return MAS


for c1 in range(c):
    partn = KC[c1][1]
    CSP(partn).append(c1)
    SPC[partn] += 1


def RGBset(n):
    k = -(math.floor(1 - (8 * n + 1) ** 0.5 / 2))
    RGBset1 = []
    if k > 1:
        x = 255 // (k - 1)
        for i in range(k):
            RGBset1.append(i * x)
        return RGBset1


def CColorSet(mat):
    n = len(mat)
    nRGB = RGBset(n)
    k = len(nRGB) - 1
    RGB = []
    for R in range(k, -1):
        if (k - R) % 2 == 0:
            for G in range(k - R + 1):
                B = k - R - G
                RGB.append(format(nRGB[R] * 16 ** 4 + nRGB[G] * 16 ** 2 + nRGB[B], 'x'))
        else:
            for G in range(k - R, -1):
                B = k - R - G
                RGB.append(format(nRGB[R] * 16 ** 4 + nRGB[G] * 16 ** 2 + nRGB[B], 'x'))
    return RGB


# 기타#
def SkDc(SkT, SCT):
    for s1 in range(s):
        for t1 in range(30):
            dat = date(t1)
            k1 = FTD[dat]
            k2 = FTD[dat + 1]
            for t2 in range(k1, k2):
                if SkT[s1][t1] == SkT[s1][t2]:
                    if SCT[s1][t1] != SCT[s1][t2]:
                        return 1
    return 0


rSCTs = []
ATT = []
stds = []
CPTs = []
rSCT = []


def columize(mat):
    return mat.reshape(-1, 1)


# highest priority control codes#
control = 1
keep = 0
back = 0
out = 0
# highest priority control codes#


# main variables define code#
while control == 1:
    CC = zCC
    for s1 in range(s):
        S1 = cSC[s1]
        CC += columize(S1) @ S1
    for c1 in range(c):
        for c2 in range(c):
            if CC[c1][c2] == 0:
                CC[c1][c2] = s
                CC[c2][c1] = s
    GSC = []
    GS = []
    for s1 in range(s):
        S2 = cSC[s1]
        if min(CC - columize(S2) @ S2) == 0:
            GS.append(s1)
            GSC.append(S2)
        else:
            CC -= columize(S2) @ S2
    g = len(GS)
    GSCP = np.ones(g)
    for g1 in range(g):
        for c1 in range(c):
            if GSC[g1][c1] == 1:
                GSCP[g1] *= Prime[c1]
    wc = 0
    cN = tN = gN = np.zeros(5)
    Td = Cd = Tm = Cm = Tq = Cq = Tt = Ct = Tp = Cp = []
    WG = RTL = []
    RTTypeT = RTTypeD = RTTypeQ = RTTypeM = RTTypeP = []
    RGSTP = np.ones((1, g))
    RCpTpS = np.ones((1, len(FCS)))
    TType = [0, 3, 6, 8, 11, 0, 0, 0, 0]
    CkT = CPT = np.ones((c, 30))
    CCT = np.zeros((c, 30))

    # Dcode#
    while cN[0] < d:
        RTL.append(do(TType))
        while tN[0] < RTL[-1] + 1:
            gN[0] = 0
            if cN[0] == d:
                del RTL[-1]
                WG = []
                wc = tN[1] = cN[1] = 0
                print("D clear")

                # Mcode#
                while cN[1] < m:
                    RTL.append(mo(TType))
                    while tN[1] < RTL[-1] + 1:
                        gN[1] = 0
                        if cN[1] == m:
                            del RTL[-1]
                            WG = []
                            wc = tN[2] = cN[2] = 0
                            print("M clear")

                            # Qcode#
                            while cN[2] < q:
                                RTL.append(qu(TType))
                                while tN[2] < RTL[-1] + 1:
                                    gN[2] = 0
                                    if cN[2] == q:
                                        del RTL[-1]
                                        WG = []
                                        wc = tN[3] = cN[3] = 0
                                        print("Q clear")

                                        # Tcode#
                                        while cN[3] < t:
                                            RTL.append(tr(TType))
                                            while tN[3] < RTL[-1] + 1:
                                                gN[3] = 0
                                                if cN[0] == t:
                                                    del RTL[-1]
                                                    WG = []
                                                    wc = tN[4] = cN[4] = 0
                                                    print("T clear")

                                                    # Pcode#
                                                    while cN[4] < p:
                                                        RTL.append(pa(TType))
                                                        while tN[4] < RTL[-1] + 1:
                                                            gN[4] = 0
                                                            if cN[4] == p:
                                                                del RTL[-1]
                                                                print("P clear")

                                                                # printcode#
                                                                for d1 in range(d):
                                                                    k1 = D[d1]
                                                                    k2 = DO(RTTypeD[d1])[Td[d1]]
                                                                    k3 = Prime[k1]
                                                                    k4 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = CPT[k1][k2 + 1] = k3
                                                                    CkT[k1][k2] = CkT[k1][k2 + 1] = k4
                                                                    CCT[k1][k2] = CCT[k1][k2 + 1] = k1

                                                                for m1 in range(m):
                                                                    k1 = M[m1]
                                                                    k2 = MO(RTTypeM[m1])[Tm[m1]]
                                                                    k3 = Prime[k1]
                                                                    k4 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = k3
                                                                    CkT[k1][k2] = k3
                                                                    CCT[k1][k2] = k3

                                                                for q1 in range(q):
                                                                    k1 = Q[q1]
                                                                    [k2, k3] = QU(RTTypeQ[q1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = CPT[k1][k2 + 1] = CPT[k1][k3] = \
                                                                        CPT[k1][k3 + 1] = k4
                                                                    CkT[k1][k2] = CkT[k1][k2 + 1] = CkT[k1][k3] = \
                                                                        CkT[k1][k3 + 1] = k5
                                                                    CCT[k1][k2] = CCT[k1][k2 + 1] = CCT[k1][k3] = \
                                                                        CCT[k1][k3 + 1] = k1

                                                                for t1 in range(t):
                                                                    k1 = T[t1]
                                                                    [k2, k3] = TR(RTTypeT[t1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = CPT[k1][k2 + 1] = CPT[k1][k3] = k4
                                                                    CkT[k1][k2] = CkT[k1][k2 + 1] = CkT[k1][k3] = k5
                                                                    CCT[k1][k2] = CCT[k1][k2 + 1] = CCT[k1][k3] = k1

                                                                for p1 in range(p):
                                                                    k1 = P[p1]
                                                                    [k2, k3] = PA(RTTypeP[p1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = CPT[k1][k3] = k4
                                                                    CkT[k1][k2] = CkT[k1][k3] = k5
                                                                    CCT[k1][k2] = CCT[k1][k3] = k1

                                                                SPT = cSC @ CPT
                                                                SkT = cSC @ CkT
                                                                SCT = cSC @ CCT

                                                                if SkDc(SkT, SCT) == 0:
                                                                    AT = thirtyTTRP(Mon, Tu, We, Th, Fr, SCT)
                                                                    SCTD = SCTd = SCT
                                                                    TRP = np.zeros(30)
                                                                    for i in range(30):
                                                                        TRP[AT[i]] = i
                                                                    dfSCTd = pd.DataFrame(SCTd,
                                                                                          columns=list(map(str, TRP)))
                                                                    for i in range(s):
                                                                        for j in range(30):
                                                                            L1 = list(map(str, [j]))
                                                                            SCTD[i][j] = dfSCTd.ix[i][L1[0]]

                                                                    # QDTF code#
                                                                    NIS = []
                                                                    NISD = np.zeros(7)
                                                                    for i in range(30):
                                                                        for s1 in range(s):
                                                                            c1 = SCTD[s1][i]
                                                                            if c1 < c:
                                                                                NISD[KC[c1][2]] += 1
                                                                        NIS.append(NISD)
                                                                    NISF = [NIS[0] + NIS[1], NIS[2] + NIS[3], NIS[4],
                                                                            NIS[5], NIS[6] + NIS[7], NIS[8] + NIS[9],
                                                                            NIS[13] + NIS[14], NIS[15] + NIS[16],
                                                                            NIS[17] + NIS[18], NIS[19] + NIS[20],
                                                                            NIS[24] + NIS[25], NIS[26] + NIS[27],
                                                                            NIS[28], NIS[29]]
                                                                    FASI = np.arange(0, 30)
                                                                    for k in range(7):
                                                                        if CIO(NISF[2 * k], NISF[2 * k + 1]) == 1:
                                                                            if k == 0:
                                                                                for i in range(0, 4):
                                                                                    FASI[i] = 3 - i
                                                                            elif k == 1:
                                                                                FASI[4] = 5
                                                                                FASI[5] = 4
                                                                            elif k == 2:
                                                                                for i in range(6, 10):
                                                                                    FASI[i] = 15 - i
                                                                            elif k == 3:
                                                                                for i in range(13, 17):
                                                                                    FASI[i] = 29 - i
                                                                            elif k == 4:
                                                                                for i in range(17, 21):
                                                                                    FASI[i] = 37 - i
                                                                            elif k == 5:
                                                                                for i in range(24, 28):
                                                                                    FASI[i] = 51 - i
                                                                            else:
                                                                                FASI[28] = 29
                                                                                FASI[29] = 28
                                                                    SCTDi = SCTDI = SCTD
                                                                    dfSCTDi = pd.DataFrame(SCTDi,
                                                                                           columns=list(map(str, FASI)))
                                                                    for i in range(s):
                                                                        for j in range(30):
                                                                            L2 = list(map(str, [j]))
                                                                            SCTDI[i][j] = dfSCTDi.ix[i][L2[0]]

                                                                    # QTTD code#
                                                                    DF = np.arange(0, 30)
                                                                    QTTD = [0, 0]
                                                                    for a in range(2):
                                                                        QTD = np.zeros(q)
                                                                        for i in range(q):
                                                                            ttypeq = RTTypeQ[i]
                                                                            QU = QU(ttypeq)
                                                                            [t1, t2] = QU[Tq[i]]
                                                                            date1 = date(t1)
                                                                            date2 = date(t2)
                                                                            date3 = DRP[a][date1]
                                                                            date4 = DRP[a][date2]
                                                                            qtd = abs(date3 - date4)
                                                                            QTD[i] = min(qtd, 7 - qtd)
                                                                        TTD = np.zeros(t)
                                                                        for i in range(t):
                                                                            ttypet = RTTypeT[i]
                                                                            TR = TR(ttypet)
                                                                            [t1, t2] = TR[Tt[i]]
                                                                            date1 = date(t1)
                                                                            date2 = date(t2)
                                                                            date3 = DRP[a][date1]
                                                                            date4 = DRP[a][date2]
                                                                            ttd = abs(date3 - date4)
                                                                            TTD[i] = min(ttd, 7 - ttd)
                                                                        QTTD[a] = sum(QTD) + sum(TTD)

                                                                    qttd = QTTD.index(min(QTTD))

                                                                    if qttd == 1:
                                                                        for i in range(6):
                                                                            DF[i] = 24 + i
                                                                            DF[24 + i] = i

                                                                    SCTDId = SCTDID = SCTDI
                                                                    dfSCTDId = pd.DataFrame(SCTDId,
                                                                                            columns=list(map(str, DF)))
                                                                    for i in range(s):
                                                                        for j in range(30):
                                                                            L3 = list(map(str, [j]))
                                                                            SCTDID[i][j] = dfSCTDId.ix[i][L3[0]]

                                                                    # STD code#
                                                                    CPTs.append(CPT)
                                                                    std = np.zeros(s)
                                                                    for s1 in range(s):
                                                                        sctd = SCTDID[s1]
                                                                        for t1 in range(30):
                                                                            c1 = sctd[t1]
                                                                            c2 = sctd[t1 + 1]
                                                                            std[s1] += DI[c1][c2]
                                                                        for i in range(9):
                                                                            c3 = sctd[NDS[i]]
                                                                            c4 = sctd[NDS[i] - 1]
                                                                            sctd[s1] -= DI[c3][c4]
                                                                    SAD = (sum(std)) / s
                                                                    rSCT1 = []
                                                                    for s1 in range(s):
                                                                        for i in range(30):
                                                                            rSCT1.append(SCTDID[s1][i])
                                                                    rSCT.append(SAD)
                                                                    rSCTs.append(rSCT)
                                                                    ATT.append(rSCT1)
                                                                    stds.append(std)
                                                                    TSCs.append(TSC(cSC))
                                                                back = 1
                                                            while back == 1:
                                                                tN[4] = Tp[-1] + 1
                                                                cN[4] = Cp[-1]
                                                                del Cp[-1]
                                                                del Tp[-1]
                                                                del RGSTP[-1]
                                                                del RTTypeP[-1]
                                                                del RCpTpS[-1]
                                                                del RTL[-1]
                                                                back = 0
                                                                if tN[4] == RTL[-1]:
                                                                    if not Cp:
                                                                        out = 1
                                                                        cN[4] = p
                                                                        tN[4] = RTL[-1] + 1
                                                                        gN[4] = g
                                                                        print("P fail")
                                                                    else:
                                                                        back = 1
                                                            GSTP = RGSTP[-1]
                                                            TType = RTTypeT[-1]
                                                            CpTpS = RCpTpS[-1]
                                                            while gN[4] < g:
                                                                if GSCP[gN[4]] % Prime[P[cN[4]]] == 0:
                                                                    PA1 = PA(TType)
                                                                    gstp = GSTP[gN[4]]
                                                                    [z1, z2] = PA1[tN[4]]
                                                                    if NED(gstp, [Prime[z1], Prime[z2]]) == 0:
                                                                        if P[cN[4]] > wc:
                                                                            WG = [gN[4]]
                                                                            wc = P[cN[4]]
                                                                        elif P[cN[4]] == wc:
                                                                            WG.append(gN[4])
                                                                        gN[4] = 0
                                                                        tN[4] += 1
                                                                        if tN[4] == RTL[-1]:
                                                                            gN[4] = g + 1
                                                                            back = 1
                                                                    else:
                                                                        gN[4] += 1
                                                                else:
                                                                    gN[4] += 1
                                                                if gN[4] == g:
                                                                    i = PIL(FCS, cN[4])
                                                                    PA1 = PA(TType)
                                                                    j1 = CpTpS[i]
                                                                    j2 = MCpTpS[i]
                                                                    [j3, j4] = PA1[tN[4]]
                                                                    if max(Pm(j1, j3), Pm(j1, j4)) == j2:
                                                                        gN[4] = 0
                                                                        tN[4] += 1
                                                                        if tN[4] == RTL[-1]:
                                                                            gN[4] = g + 1
                                                                            back = 1
                                                                    else:
                                                                        j5 = Prime[j3] * Prime[j4]
                                                                        CpTpS[i] = j1 * j5
                                                                        RCpTpS.append(CpTpS)
                                                                        Cp.append(cN[4])
                                                                        Tp.append(tN[4])
                                                                        for g1 in range(g):
                                                                            if GSCP[g1] % Prime[P[cN[4]]] == 0:
                                                                                GSTP[g1] *= j5
                                                                        RGSTP.append(GSTP)
                                                                        TType = PTType(TType)[tN[4]]
                                                                        RTL.append(pa(TType))
                                                                        RTTypeP.append(TType)
                                                                        cN[4] += 1
                                                                        tN[4] = 0
                                                if out == 1:
                                                    back = 1
                                                    out = 0
                                                while back == 1:
                                                    tN[3] = Tt[-1] + 1
                                                    cN[3] = Ct[-1]
                                                    del Ct[-1]
                                                    del Tt[-1]
                                                    del RGSTP[-1]
                                                    del RTTypeT[-1]
                                                    del RCpTpS[-1]
                                                    del RTL[-1]
                                                    back = 0
                                                    if tN[3] == RTL[-1]:
                                                        if not Ct:
                                                            out = 1
                                                            cN[3] = t
                                                            tN[3] = RTL[-1] + 1
                                                            gN[3] = g
                                                            print("T fail")
                                                        else:
                                                            back = 1
                                                GSTP = RGSTP[-1]
                                                TType = RTTypeP[-1]
                                                CpTpS = RCpTpS[-1]
                                                while gN[3] < g:
                                                    if GSCP[gN[3]] % Prime[T[cN[3]]] == 0:
                                                        TR1 = TR(TType)
                                                        gstp = GSTP[gN[3]]
                                                        [z1, z2] = TR1[tN[3]]
                                                        if NED(gstp, SelIO(Prime, z1, 2)) * (gstp % Prime[z2]) == 0:
                                                            if T[cN[3]] > wc:
                                                                WG = [gN[3]]
                                                                wc = T[cN[3]]
                                                            elif T[cN[3]] == wc:
                                                                WG.append(gN[3])
                                                            gN[3] = 0
                                                            tN[3] += 1
                                                            if tN[3] == RTL:
                                                                gN[3] = g + 1
                                                                back = 1
                                                        else:
                                                            gN[3] += 1
                                                    else:
                                                        gN[3] += 1
                                                    if gN[3] == g:
                                                        i = PIL(FCS, cN[3])
                                                        TR1 = TR(TType)
                                                        j1 = CpTpS[i]
                                                        j2 = MCpTpS[i]
                                                        [j3, j4] = TR1[tN[3]]
                                                        if max(Pm(j1, j3), Pm(j1, j3 + 1), Pm(j1, j4)) == j2:
                                                            gN[3] = 0
                                                            tN[3] += 1
                                                            if tN[3] == RTL[-1]:
                                                                gN[3] = g + 1
                                                                back = 1
                                                        else:
                                                            j5 = MIO(Prime, j3, 2) * Prime[j4]
                                                            CpTpS[i] = j1 * j5
                                                            RCpTpS.append(CpTpS)
                                                            Ct.append(cN[3])
                                                            Tt.append(tN[3])
                                                            for g1 in range(g):
                                                                if GSCP[g1] % Prime[T[cN[3]]]:
                                                                    GSTP[g1] *= j5
                                                            RGSTP.append(GSTP)
                                                            TType = TTType(TType)[tN[3]]
                                                            RTL.append(tr(TType))
                                                            RTTypeT.append(TType)
                                                            cN[3] += 1
                                                            tN[3] = 0
                                    if out == 1:
                                        back = 1
                                        out = 0
                                    while back == 1:
                                        tN[2] = Tq[-1] + 1
                                        cN[2] = Cq[-1]
                                        del Cq[-1]
                                        del Tq[-1]
                                        del RGSTP[-1]
                                        del RTTypeQ[-1]
                                        del RCpTpS[-1]
                                        del RTL[-1]
                                        back = 0
                                        if tN[2] == RTL[-1]:
                                            if not Cq:
                                                out = 1
                                                cN[2] = q
                                                tN[2] = RTL[-1] + 1
                                                gN[2] = g
                                                print("Q fail")
                                            else:
                                                back = 1
                                    GSTP = RGSTP[-1]
                                    TType = RTTypeQ[-1]
                                    CpTpS = RCpTpS[-1]
                                    while gN[2] < g:
                                        if GSCP[gN[2]] % Prime[Q[cN[2]]] == 0:
                                            QU1 = QU(TType)
                                            gstp = GSTP[gN[2]]
                                            [z1, z2] = QU1[tN[2]]
                                            if NED(gstp, SelIO(Prime, z1, 2)) * NED(gstp, SelIO(Prime, z2, 2)) == 0:
                                                if Q[cN[2]] > wc:
                                                    WG = [gN[2]]
                                                    wc = Q[cN[2]]
                                                elif Q[cN[2]] == wc:
                                                    WG.append(gN[2])
                                                gN[2] = 0
                                                tN[2] += 1
                                                if tN[2] == RTL[-1]:
                                                    gN[2] = g + 1
                                                    back = 1
                                            else:
                                                gN[2] += 1
                                        else:
                                            gN[2] += 1
                                        if gN[2] == g:
                                            i = PIL(FCS, cN[2])
                                            QU1 = QU(TType)
                                            j1 = CpTpS[i]
                                            j2 = MCpTpS[i]
                                            j4 = QU1[tN[2]][1]
                                            j3 = QU1[tN[2]][0]
                                            if max(Pm(j1, j3), Pm(j1, j3 + 1), Pm(j1, j4), Pm(j1, j4 + 1)) == j2:
                                                gN[2] = 0
                                                tN[2] += 1
                                                if tN[2] == RTL[-1]:
                                                    gN[2] += 1
                                                    back = 1
                                            else:
                                                j5 = MIO(Prime, j3, 2) * MIO(Prime, j4, 2)
                                                CpTpS[i] = j1 * j5
                                                RCpTpS.append(CpTpS)
                                                Cq.append(cN[2])
                                                Tq.append(tN[2])
                                                for g1 in range(g):
                                                    if GSCP[g1] % Prime[Q[cN[2]]] == 0:
                                                        GSTP[g1] *= j5
                                                RGSTP.append(GSTP)
                                                TType = QTType(TType)[tN[2]]
                                                RTL.append(qu(TType))
                                                RTTypeQ.append(TType)
                                                cN[2] += 1
                                                tN[2] = 0
                        if out == 1:
                            back = 1
                            out = 0
                        while back == 1:
                            tN[1] = Tm[-1] + 1
                            cN[1] = Cm[-1]
                            del Cm[-1]
                            del Tm[-1]
                            del RGSTP[-1]
                            del RTTypeM[-1]
                            del RCpTpS[-1]
                            del RTL[-1]
                            back = 0
                            if tN[1] == RTL[-1]:
                                if not Cm:
                                    out = 1
                                    cN[1] = m
                                    tN[1] = RTL[-1] + 1
                                    gN[1] = g
                                    print("M fail")
                                else:
                                    back = 1
                        GSTP = RGSTP[-1]
                        TType = RTTypeM[-1]
                        CpTpS = RCpTpS[-1]
                        while gN[1] < g:
                            if GSCP[gN[1]] % Prime[M[cN[1]]] == 0:
                                MO1 = MO(TType)
                                gstp = GSTP[gN[1]]
                                z1 = MO1[tN[1]]
                                if gstp % Prime[z1] == 0:
                                    if M[cN[1]] > wc:
                                        WG = [gN[1]]
                                        wc = M[cN[1]]
                                    elif M[cN[1]] == wc:
                                        WG.append(gN[1])
                                    gN[1] = 0
                                    tN[1] += 1
                                    if tN[1] == RTL[-1]:
                                        gN[1] = g + 1
                                        back = 1
                                else:
                                    gN[1] += 1
                            else:
                                gN[1] += 1
                            if gN[1] == g:
                                i = PIL(FCS, cN[1])
                                MO1 = MO(TType)
                                j1 = CpTpS[i]
                                j2 = MCpTpS[i]
                                j3 = MO1[tN[1]]
                                if Pm(j1, j3) == j2:
                                    gN[1] = 0
                                    tN[1] += 1
                                    if tN[1] == RTL[-1]:
                                        gN[1] = g + 1
                                        back = 1
                                else:
                                    CpTpS[i] = j1 * Prime[j3]
                                    RCpTpS.append(CpTpS)
                                    Cm.append(cN[1])
                                    Tm.append(tN[1])
                                    for g1 in range(g):
                                        if GSCP[g1] % Prime[M[cN[1]]] == 0:
                                            GSTP[g1] *= Prime[j3]
                                    RGSTP.append(GSTP)
                                    TType = CTType(1, MO1[tN[1]], TType)
                                    RTL.append(mo(TType))
                                    RTTypeM.append(TType)
                                    cN[1] += 1
                                    tN[1] = 0
            if out == 1:
                back = 1
                out = 0
            while back == 1:
                tN[0] = Td[-1] + 1
                cN[0] = Cd[-1]
                del Cd[-1]
                del Td[-1]
                del RGSTP[-1]
                del RTTypeD[-1]
                del RCpTpS[-1]
                del RTL[-1]
                back = 0
                if tN[0] == RTL[-1]:
                    if not Cd:
                        control = 0
                        cN[0] = d
                        tN[0] = RTL[-1] + 1
                        gN[0] = g
                        print("!!!Control Code Activated!!!")
                    else:
                        back = 1
            GSTP = RGSTP[-1]
            TType = RTTypeD[-1]
            CpTpS = RCpTpS[-1]
            while gN[0] < g:
                if GSCP[gN[0]] % Prime[D[cN[0]]] == 0:
                    DO1 = DO(TType)
                    gstp = GSTP[gN[0]]
                    z1 = DO1[tN[0]]
                    if NED(gstp, SelIO(Prime, z1, 2)) == 0:
                        if D[cN[0]] > wc:
                            WG = [gN[0]]
                            wc = D[cN[0]]
                        elif D[cN[0]] == wc:
                            WG.append(gN[0])
                        gN[0] = 0
                        tN[0] += 1
                        if tN[0] == RTL[-1]:
                            gN[0] = g + 1
                            back = 1
                    else:
                        gN[0] += 1
                else:
                    gN[0] += 1
                if gN[0] == g:
                    i = PIL(FCS, cN[0])
                    DO1 = DO(TType)
                    j1 = CpTpS[i]
                    j2 = MCpTpS[i]
                    j3 = DO1[tN[0]]
                    if max(Pm(j1, j3), Pm(j1, j3 + 1)) == j2:
                        gN[0] = 0
                        tN[0] += 1
                        if tN[0] == RTL[-1]:
                            gN[0] = g + 1
                            back = 1
                        else:
                            j4 = MIO(Prime, j3, 2)
                            CpTpS[i] = j1 * j4
                            RCpTpS.append(CpTpS)
                            Cd.append(cN[0])
                            Td.append(tN[0])
                            for g1 in range(g):
                                if GSCP[g1] % Prime[D[cN[0]]] == 0:
                                    GSCP[g1] = GSCP[g1] * j4
                                    RGSTP.append(GSTP)
                                    TType = CTType(0, DO1[tN[0]], TType)
                                    RTL.append(do(TType))
                                    RTTypeD.append(TType)
                                    cN[0] += 1
                                    tN[0] = 0

    if control == 0:
        if not list({wc} - set(LCS)):
            LCS2 = LCS[0]
            sub = LCS2.index(wc)
            for ws in range(len(WG)):
                cSC[GS[WG[ws]]][wc] = 0
                cSC[GS[WG[ws]]][FCS[sub]] = 1
                print(GS[WG[ws]], ":", wc, "->", FCS[sub])
        else:
            for ws in range(len(WG)):
                cSC[GS[WG[ws]]][wc] = 0
                cSC[GS[WG[ws]]][wc + 1] = 1
                print(GS[WG[ws]], ":", wc, "->", wc + 1)
        if dfSC == pd.DataFrame(cSC, columns=CCS(c)):
            if not rSCTs:
                print("ERROR")
            else:
                keep = 1
        else:
            control = 1
            print("control code deactivated")

if keep == 1:
    rSCTDs = []
    dfrSCTds = pd.DataFrame(rSCTs, columns=np.arange(0, 30 * s).append('SAD'))
    l = len(dfrSCTds.index)
    dfrSCTds['T'] = np.arange(0, l)
    dfrSCTds.sort_values('SAD', ascending=True)
    TtT1 = dfrSCTds.iloc['T']
    for i in range(l):
        for j in range(30 * s):
            L4 = list(map(str, [j]))
            rSCTDs[i][j] = dfrSCTds.ix[i][L4[0]]
    for i in range(l):
        OTT = np.array(ATT[TtT1[i]])
        rOTT2 = OTT.reshape(s, 30)
        rOTT = rOTT2.tolist()
        CPT0 = CPTs[TtT1[i]]
        for u in range(12):
            pn = SPC[u]
            le = 40 / pn
            PART = CSP(u)
            for j in range(pn):
                k = PART[j]
                for t1 in range(30):
                    if CPT0[k][t1] > 0:
                        RGB2 = CColorSet(PART)
                        plt.fill([t1, t1, t1 + 1, t1 + 1],
                                 [le * (j + 0.25), le * (j + 0.75), le * (j + 0.75), le * (j + 0.25)],
                                 color='#' + '0' * (6 - len(RGB2[j])) + str(RGB2[j]), alpha=0.5)
            plt.title(GT[u])
            plt.show()

        plt.bar(C, TSCs[TtT1[i]], color='#e35f62')
        plt.xticks(C, C)
        plt.show()
        std2 = stds[TtT1[i]]
        s1std = s2std = std2
        dfs1std = pd.DataFrame(s1std, columns=list(map(str, StS)))
        for j in range(s):
            L5 = list(map(str, [j]))
            s2std[j] = dfs1std.ix[0][L5[0]]
        plt.rcParams["figure.figsize"] = (40, 40)
        plt.plot(np.arange(s), s2std)
        plt.xlabel('students')
        plt.ylabel('average distance')
        plt.show()
    print("What Time Table?")
    v = int(input())

    # PTTprint code#
