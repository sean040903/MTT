import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

# input 2-dimension lists#
"""print("input DPN")
DPN = list(map(int, input().split()))"""
DPN = [6, 7, 4, 6, 6]
"""print("input s")
s = int(input())"""
s = 263
S = np.arange(s)
"""print("input l")
l = int(input())"""
l = 62
L = np.arange(l)
print("input SL")
SL = np.array([list(map(int, input().split())) for _ in range(0, s)])  # Student's classes#
print("input KL")
KL = np.array([list(map(int, input().split())) for _ in
               range(0, l)])  # Kind of Class, KC[i][0]: 시수 종류, KC[i][1]: 과목 종류, KC[i][2]:과목 중요도#
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
"""print("input MCpTpS")
MCpTpS = list(map(int, input().split()))  # Max of Class per Time per Subjects#"""
MCpTpS = [4, 4, 4, 4, 2, 4, 4, 4, 4, 1, 1, 4, 4, 2, 4, 4, 2, 4, 4, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1,
          1, 1, 1, 2, 2, 1, 1, 2, 1, 1, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1]
"""print("input NCL")
NCL = list(map(int, input().split()))"""
NCL = [8, 8, 7, 1, 2, 4, 6, 8, 8, 2, 1, 8, 8, 2, 8, 8, 1, 8, 2, 2, 8, 3, 3, 3, 8, 8, 8, 1, 3, 8, 8, 4, 4, 4, 1, 8, 4, 4,
       4, 2, 2, 7, 8, 8, 2, 2, 2, 2, 2, 1, 10, 10, 8, 1, 1, 2, 3, 1, 3, 1, 2, 1]
print("input DI")
DI = np.array([list(map(float, input().split())) for _ in range(0, l)])  # Distance#

thisplace = 0


# 행렬-원소 선택 함수들#
def PIL(mat, n):
    for i1 in range(0, len(mat)):
        if n < mat[i1]:
            return i1 - 1
    return len(mat) - 1


def NED(n, mat):
    for i2 in range(0, len(mat)):
        if n % mat[i2] == 0:
            return 0
    return 1


# DPN 활용#
daily = max(DPN)
weekly = len(DPN)

FPD = [0]
for i3 in range(0, weekly - 1):
    FPD.append(FPD[-1] + DPN[i3])


def dates(mat):
    Dates = []
    for i4 in range(0, len(mat)):
        Dates.append(PIL(FPD, mat[i4]))
    return Dates


def date(n):
    return PIL(FPD, n)


def CTType(k, SelP, mat):
    mat2 = mat[:]
    for i5 in range(0, weekly):
        if mat[i5] < DMEP[i5]:
            if len(SelP) == 1:
                if SelP[0] == EP[PIL(EP, mat[i5])]:
                    mat2[i5] += 2
            else:
                if SelP[0] == EP[PIL(EP, mat[i5])]:
                    mat2[i5] += 2
                if SelP[1] == EP[PIL(EP, mat[i5])]:
                    mat2[i5] += 2
    DATES = dates(SelP)[:]
    bfs = format(26 - k, 'b')
    for j in range(0, len(SelP)):
        if not list({DATES[j]} - set(DCD)):
            mat2[weekly + DCD.index(DATES[j])] = 1
        if bfs[-j - 1] == 1:
            if not list({SelP[j]} - set(EP) - set(VEP)):
                mat2[-1] = BFN(np.union1d(np.array(NZBF(mat[-1])), np.array([len(EP) - PIL(EP, SelP[j])])))
    if len(SelP) > 1:
        if bfs[-1] == bfs[-2]:
            if not list({tuple(DATES)} - {tuple(i) for i in SDPD}):
                i0 = DCD.index(DATES[0]) + weekly
                if mat[i0] == 0:
                    if SelP[0] + sum(SelIO(DPN, DATES[0], DATES[1] - DATES[0])) == SelP[1]:
                        mat2[i0] = 0
    return mat2


# making classes#
FCL = [0]
ECL = []
for i6 in range(0, l):
    a1 = FCL[-1]
    a2 = NCL[i6]
    FCL.append(a1 + a2)
    ECL.append(a1 + a2 - 1)

LTS = [0] * l
for i7 in range(0, s):
    LTS += SL[i7]


def DSL(n):
    dsl = []
    for i8 in range(0, NCL[n]):
        dsl.append(np.floor(LTS[n] * (i8 + 1) / NCL[n]) - np.floor(LTS[n] * i8 / NCL[n]))
    return dsl


def MDI(mat1, mat2):
    for i9 in range(0, len(mat1)):
        if mat1[i9] != mat2[i9]:
            return i9
    return len(mat1)


c = sum(NCL)
C = np.arange(c)

SC = np.zeros((s, c))
for l1 in range(0, l):
    MS = np.zeros(NCL[l1])
    for s1 in range(0, s):
        if SL[s1][l1] == 1:
            SC[s1][FCL[l1] + MDI(MS, DSL(l1))] = 1
            MS[MDI(MS, DSL(l1))] += 1

KC = []
for c1 in range(0, c):
    KC.append(KL[PIL(FCL, c1) - 1])

# SC sorting#
cSC = SC[:]


def CCS(n):
    return list(map(str, np.arange(0, n)))


TCS = np.zeros(s)
for s1 in range(0, s):
    TCS[s1] = sum(SC[s1])

dfSC = pd.DataFrame(SC, columns=CCS(c))
dfSC['TCS'] = TCS
dfSC['S'] = np.arange(0, s)
dfSC.sort_values('TCS', ascending=True)
StS = np.zeros(s)
for i in range(0, s):
    StS[i] = dfSC.iloc[i]['S']
dfSC.drop('TCS', axis=1, inplace=True)
dfSC.drop('S', axis=1, inplace=True)
dfSC = dfSC.reset_index(drop=True)

for s1 in range(0, s):
    for c1 in range(0, c):
        l1 = list(map(str, [c1]))
        cSC[s1][c1] = dfSC.iloc[s1][l1[0]]


def TotSC(mat):
    TotSC = np.zeros(c)
    for s1 in range(0, s):
        TotSC += [mat[s1]]
    return TotSC


TotSCs = []

# 반 분류#
Q = []
T = []
M = []
D = []
P = []
QU = []
TR = []
PA = []
m = 0
d = 0
q = 0
t = 0
p = 0
for c1 in range(c):
    if KC[c1][0] == 12:
        T.append(c1)
        t += 1
    elif KC[c1][0] == 2:
        D.append(c1)
        d += 1
    elif KC[c1][0] == 11:
        P.append(c1)
        p += 1
    elif KC[c1][0] == 1:
        M.append(c1)
        m += 1
    else:
        Q.append(c1)
        q += 1


# 소수,진법 변환 관련 함수들#
def Pm(n1, n2):
    multiply = 0
    while n1 % (Prime[n2] ** multiply) == 0:
        multiply += 1
        return multiply - 1


def NBF(n):
    nbf = format(n, 'b')
    return list(map(int, nbf))


def NZBF(n):
    nzbf = []
    nbf = format(n, 'b')
    for i in range(0, len(nbf)):
        if nbf[i] == '1':
            nzbf.append(i)
    return nzbf


def BFN(mat):
    n = 0
    for i in range(0, len(mat)):
        n += 2 ** mat[i]
    return n


# 행렬 연속 몇개 선택 함수들#
def MIO(mat, i, n):  # Multiply In Order#
    mio = 1
    for j in range(i, i + n):
        mio *= mat[j]
    return mio


def SelIO(mat, i, n):
    selio = []
    for j in range(i, i + n):
        selio.append(mat[j])
    return selio


def C2N(x, y):
    c2n = []
    for i in range(x, y, 2):
        c2n.append(i)
    return c2n


# 행렬-행렬관계 함수#
def CIO(mat1, mat2):
    for i in range(0, len(mat1)):
        if mat1[i] > mat2[i]:
            return 1
        elif mat2[i] > mat1[i]:
            return 2
    return 0


# 교시교환 관련 함수들#
def LW1P(da, mat):
    a = FPD[da]
    tw = np.zeros((2, 2))
    for i in range(0, 2):
        for j in range(0, 2):
            for s1 in range(0, s):
                c1 = mat[s1][a + 1 - i]
                c2 = mat[s1][a + 2 + j]
                if max(c1, c2) < c:
                    tw[i][j] += DI[c1][c2]
    x, y = np.where(tw == np.min(tw))
    mwp = [a + x, a + 1 - x, a + 2 + y, a + 3 - y]
    if DPN[da] % 2 == 1:
        mwp.append(a + 4)
    return mwp


def LW2P(da, mat):
    a = FPD[da]
    tw = np.zeros((3, 2, 2))
    for i in range(0, 3):
        for j in range(0, 2):
            for k in range(0, 2):
                for s1 in range(0, s):
                    c1 = mat[s1][a + 2 * (i % 3) + 1 - i]
                    c2 = mat[s1][a + 2 * ((d + 1) % 3) + j]
                    if max(c1, c2) < c:
                        tw[i][j][k] += DI[c1][c2]
    x, y, z = np.where(tw == np.min(tw))
    x0 = 2 * (x % 3)
    x1 = 2 * ((x + 1) % 3)
    x2 = 2 * ((x + 2) % 3)
    return [a + x0 + y, a + x0 + 1 - y, a + x1 + z, a + x1 + 1 - z, a + x2, a + x2 + 1]


def LW3P(da, mat):
    a = FPD[da]
    tw = np.zeros((3, 2, 2, 2))
    for i in range(0, 3):
        for j in range(0, 2):
            for k in range(0, 2):
                for o in range(0, 2):
                    for s1 in range(0, s):
                        c1 = mat[s1][a + 2 * (i % 3) + 1 - j]
                        c2 = mat[s1][a + 2 * ((i + 1) % 3) + k]
                        c3 = mat[s1][a + 2 * ((i + 2) % 3) + 1 - o]
                        c4 = mat[s1][a + 6]
                        if max(c1, c2) < c:
                            tw[i][j][k][o] += DI[c1][c2]
                        if max(c3, c4) < c:
                            tw[i][j][k][o] += DI[c3][c4]
    x, y, z, w = np.where(tw == np.min(tw))
    x0 = 2 * (x % 3)
    x1 = 2 * ((x + 1) % 3)
    x2 = 2 * ((x + 2) % 3)
    return [a + x0 + y, a + x0 + 1 - y, a + x1 + z, a + x1 + 1 - z, a + x2 + w, a + x2 + 1 - w, a + 6]


def LWP(mat):
    tmwp = []
    for i in range(0, weekly):
        if DPN[i] < 6:
            tmwp += LW1P(i, mat)
        elif DPN[i] == 6:
            tmwp += LW2P(i, mat)
        else:
            tmwp += LW3P(i, mat)
    return tmwp


# EP,VEP관련 함수들#

def EPs(mat):
    eps = []
    for i in range(0, len(mat)):
        eps.append(EP[mat[i]])
    return eps


def VEPs(mat):
    veps = []
    for i in range(0, len(mat)):
        veps.append(VEP[mat[i]])
    return veps


def Even(mat):
    eap = []
    for i1 in range(0, weekly):
        if not {i1} - set(CAD):
            if mat[SDPI[CAD.index(i1)]] == 1:
                eap += C2N(FPD[i1], mat[i1] + 2)
        else:
            eap += C2N(FPD[i1], mat[i1] + 2)
    return eap


def Odd(mat):
    oap = []
    for i10 in range(0, weekly):
        if not {i10} - set(CAD):
            if mat[SDPI[CAD.index(i10)]] == 1:
                oap += C2N(FPD[i10], mat[i10] + 2)
                if DPN[i10] % 2 == 1:
                    oap.append(NEP[NEPI.index(i10)])
        else:
            if FPD[i10] == mat[i10]:
                oap.append(mat[i10])
            else:
                oap += C2N(FPD[i10], mat[i10] + 2)
            if DPN[i10] % 2 == 1:
                oap.append(NEP[NEPI.index(i10)])
    for i11 in range(0, len(NZBF(mat[-1]))):
        oap.append(VEP[NZBF(mat[-1])[i11]])
    return oap


# BI,MO,QU,TR,PA 관련#
def DO(mat):
    return Even(mat)


def do(mat):
    return len(DO(mat))


def MO(mat):
    return Odd(mat)


def mo(mat):
    return len(MO(mat))


def TR(mat):
    TR1 = []
    L1 = Even(mat)
    do1 = do(mat)
    for t1 in range(do1):
        mat1 = CTType(2, [L1[t1]], mat)
        L2 = Odd(mat1)
        mo1 = mo(mat1)
        for t2 in range(mo1):
            if PIL(FPD, L1[t1]) != PIL(FPD, L2[t2]):
                TR1.append([L1[t1], L2[t2]])
    return TR1


def tr(mat):
    tr1 = 0
    L1 = Even(mat)
    do1 = do(mat)
    for t1 in range(do1):
        mat1 = CTType(2, [L1[t1]], mat)
        L2 = Odd(mat1)
        mon = mo(mat1)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                tr1 += 1
    return tr1


def QU(mat):
    QU1 = []
    L1 = Even(mat)
    do1 = do(mat)
    for t1 in range(do1):
        mat1 = CTType(2, [L1[t1]], mat)
        L2 = Even(mat1)
        do2 = do(mat1)
        for t2 in range(do2):
            if date(L2[t2]) > date(L1[t1]):
                QU1.append([L1[t1], L2[t2]])
    return QU1


def qu(mat):
    qu1 = 0
    L1 = Even(mat)
    do1 = do(mat)
    for t1 in range(do1):
        mat1 = CTType(2, [L1[t1]], mat)
        L2 = Even(mat1)
        do2 = do(mat1)
        for t2 in range(do2):
            if date(L2[t2]) > date(L1[t1]):
                qu1 += 1
    return qu1


def PA(mat):
    PA1 = []
    L1 = Odd(mat)
    mon1 = mo(mat)
    for t1 in range(mon1):
        mat1 = CTType(1, [L1[t1]], mat)
        L2 = Odd(mat1)
        mon2 = mo(mat1)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                PA1.append([L1[t1], L2[t2]])
    return PA1


def pa(mat):
    pa1 = 0
    L1 = Odd(mat)
    mon1 = mo(mat)
    for t1 in range(mon1):
        mat1 = CTType(1, [L1[t1]], mat)
        L2 = Odd(mat1)
        mon2 = mo(mat1)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                pa1 += 1
    return pa1


# 그래프 색깔#
SPC = np.zeros(12)
GT = ['Mathematics', 'Physics', 'Chemistry', 'Computer Science', 'Life Science', 'Earth Science', 'Convergence Science',
      'Science Experiment', 'Creative Convergence Special Lecture', 'Language', 'Society', 'Music-Art-Sport']
Math = []
Phy = []
Che = []
CoSc = []
LiSc = []
EaSc = []
ConS = []
ScEx = []
CCSL = []
Lan = []
Soc = []
MAS = []


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
                Blue = k - R - G
                RGB.append(format(nRGB[R] * 16 ** 4 + nRGB[G] * 16 ** 2 + nRGB[Blue], 'x'))
        else:
            for G in range(k - R, -1):
                Blue = k - R - G
                RGB.append(format(nRGB[R] * 16 ** 4 + nRGB[G] * 16 ** 2 + nRGB[Blue], 'x'))
    return RGB


PTTcolorset = ['mistyrose', 'lightcoral', 'coral', 'darkorange', 'yellow', 'greenyellow', 'chartreuse', 'palegreen',
               'aquamarine', 'paleturquoise', 'cyan', 'deepskyblue', 'thistle', 'violet', 'hotpink']


# 기타#
def SkDc(SkT, SCT):
    for s1 in range(s):
        for t1 in range(wpn):
            dat = date(t1)
            k1 = FPD[dat]
            k2 = FPD[dat + 1]
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

# TType 설정 코드#
IDP = np.union1d(np.array(FPD), np.array(FPD) + 4).tolist()
wpn = sum(DPN)
DEPN = []
for i in range(weekly):
    DEPN.append(DPN[i] // 2)
EP = []
for i in range(weekly):
    EP.append(FPD[i])
    for j in range(DEPN[i] - 1):
        EP.append(EP[-1] + 2)
VEP = (np.array(EP) + 1).tolist()
NEP = np.setdiff1d(np.arange(wpn), np.union1d(EP, VEP)).tolist()
NEPI = dates(NEP)
DMEP = []
for i in range(weekly):
    DMEP.append(FPD[i] + 2 * DEPN[i] - 2)
SUDPI = []
SDPI = []
SUDPD = []
SDPD = []
DCD = []
CAD = []
TType = FPD[:]
for i in range(weekly - 1):
    j = i
    fi = 1
    while (weekly - j) * fi > 1:
        j += 1
        if DPN[i] == DPN[j]:
            if DPN[i] % 2 == 1:
                SUDPI.append(len(TType))
                SUDPD.append(j)
            SDPI.append(len(TType))
            TType.append(0)
            SDPD.append([i, j])
            DCD.append(i)
            CAD.append(j)
            fi = 0
TType.append(0)
cTTyped = TType[:]
ADS = [np.arange(weekly).tolist()]
nds = 1
done1 = 1
while done1 == 1:
    i = 0
    j = 0
    re = 1
    while re * (nds - i) > 0:
        ADS1 = ADS[i]
        while re * (len(DCD) - j) > 0:
            k0 = DCD[j]
            k1 = CAD[j]
            ADS2 = ADS1
            ADS2[k0] = ADS1[k1]
            ADS2[k1] = ADS1[k0]
            if list({tuple(ADS2)} - {tuple(i) for i in ADS}):
                ADS.append(ADS2)
                re = 0
                nds += 1
            else:
                j += 1
        if re == 1:
            i += 1
    if re == 1:
        done1 = 0

# highest priority control codes#
control = 1
keep = 0
back = 0
out = 0
# highest priority control codes#


# main variables define code#
while control == 1:
    CC = np.zeros((c, c))
    for s1 in range(s):
        SS = np.zeros((c, c))
        SS[s1] = SC[s1][:]
        SS1 = SS
        SS2 = SS.T
        SS3 = SS2 @ SS1
        CC += SS3
    g = 0
    GS = []
    GSC = []
    GSCP = []
    for s2 in range(s):
        SS = np.zeros((c, c))
        SS[s2] = SC[s2][:]
        SS1 = SS
        SS2 = SS.T
        SS3 = SS2 @ SS1
        if np.min(CC - SS3) == 0:
            GS.append(s2)
            g += 1
            GSC.append(SC)
            gscp = 1
            for c1 in range(c):
                if SC[s2][c1] == 1:
                    gscp = gscp * Prime[c1]
            GSCP.append(gscp)
        else:
            CC -= SS3
    wc = 0
    cN = [0, 0, 0, 0, 0]
    tN = [0, 0, 0, 0, 0]
    gN = [0, 0, 0, 0, 0]
    Td = []
    Cd = []
    Tm = []
    Cm = []
    Tq = []
    Cq = []
    Tt = []
    Ct = []
    Tp = []
    Cp = []
    WG = []
    RTL = []
    RTTypeD = [cTTyped]
    RTTypeM = [[]]
    RTTypeQ = [[]]
    RTTypeT = [[]]
    RTTypeP = [[]]
    RGSTP = [[1] * g]
    RCpTpS = [[1] * len(FCL)]
    CkT = np.ones((c, wpn))
    CPT = np.ones((c, wpn))
    CCT = np.zeros((c, wpn))
    """print("checkpoint1")"""
    print(1)
    while cN[0] < d:
        RTL.append(do(cTTyped))
        while tN[0] < RTL[-1] + 1:
            gN[0] = 0
            if cN[0] == d:
                del RTL[-1]
                WG = []
                wc = 0
                tN[1] = 0
                cN[1] = 0
                """print("checkpoint2")"""
                print("D clear")
                RTTypeM=[RTTypeD[-1][:]]
                while cN[1] < m:
                    RTL.append(mo(cTTyped))
                    while tN[1] < RTL[-1] + 1:
                        gN[1] = 0
                        if cN[1] == m:
                            del RTL[-1]
                            WG = []
                            wc = 0
                            tN[2] = 0
                            cN[2] = 0
                            print("M clear")
                            RTTypeQ = [RTTypeM[-1][:]]
                            while cN[2] < q:
                                RTL.append(qu(cTTyped))
                                while tN[2] < RTL[-1] + 1:
                                    gN[2] = 0
                                    if cN[2] == q:
                                        del RTL[-1]
                                        WG = []
                                        wc = 0
                                        tN[3] = 0
                                        cN[3] = 0
                                        print("Q clear")
                                        RTTypeT = [RTTypeQ[-1][:]]
                                        while cN[3] < t:
                                            RTL.append(tr(cTTyped))
                                            while tN[3] < RTL[-1] + 1:
                                                gN[3] = 0
                                                if cN[0] == t:
                                                    del RTL[-1]
                                                    WG = []
                                                    wc = 0
                                                    tN[4] = 0
                                                    cN[4] = 0
                                                    print("T clear")
                                                    RTTypeP = [RTTypeT[-1][:]]
                                                    while cN[4] < p:
                                                        RTL.append(pa(cTTyped))
                                                        while tN[4] < RTL[-1] + 1:
                                                            gN[4] = 0
                                                            if cN[4] == p:
                                                                del RTL[-1]
                                                                print("P clear")
                                                                for d1 in range(d):
                                                                    k1 = D[d1]
                                                                    k2 = DO(RTTypeD[d1])[Td[d1]]
                                                                    k3 = Prime[k1]
                                                                    k4 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = k3
                                                                    CPT[k1][k2 + 1] = k3
                                                                    CkT[k1][k2] = k4
                                                                    CkT[k1][k2 + 1] = k4
                                                                    CCT[k1][k2] = k1
                                                                    CCT[k1][k2 + 1] = k1
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
                                                                    k2, k3 = QU(RTTypeQ[q1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = k4
                                                                    CPT[k1][k2 + 1] = k4
                                                                    CPT[k1][k3] = k4
                                                                    CPT[k1][k3 + 1] = k4
                                                                    CkT[k1][k2] = k5
                                                                    CkT[k1][k2 + 1] = k5
                                                                    CkT[k1][k3] = k5
                                                                    CkT[k1][k3 + 1] = k5
                                                                    CCT[k1][k2] = k1
                                                                    CCT[k1][k2 + 1] = k1
                                                                    CCT[k1][k3] = k1
                                                                    CCT[k1][k3 + 1] = k1
                                                                for t1 in range(t):
                                                                    k1 = T[t1]
                                                                    k2, k3 = TR(RTTypeT[t1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = k4
                                                                    CPT[k1][k2 + 1] = k4
                                                                    CPT[k1][k3] = k4
                                                                    CkT[k1][k2] = k5
                                                                    CkT[k1][k2 + 1] = k5
                                                                    CkT[k1][k3] = k5
                                                                    CCT[k1][k2] = k1
                                                                    CCT[k1][k2 + 1] = k1
                                                                    CCT[k1][k3] = k1
                                                                for p1 in range(p):
                                                                    k1 = P[p1]
                                                                    k2, k3 = PA(RTTypeP[p1])
                                                                    k4 = Prime[k1]
                                                                    k5 = Prime[KC[k1][1]]
                                                                    CPT[k1][k2] = k4
                                                                    CPT[k1][k3] = k4
                                                                    CkT[k1][k2] = k5
                                                                    CkT[k1][k3] = k5
                                                                    CCT[k1][k2] = k1
                                                                    CCT[k1][k3] = k1
                                                                SPT = cSC @ CPT
                                                                SkT = cSC @ CkT
                                                                SCT = cSC @ CCT
                                                                """print("checkpoint7")"""
                                                                if SkDc(SkT, SCT) == 0:
                                                                    AT = LWP(SCT)[:]
                                                                    SCTD = SCT[:]
                                                                    SCTd = SCT[:]
                                                                    TRP = np.zeros(wpn)
                                                                    for i in range(wpn):
                                                                        TRP[AT[i]] = i
                                                                    dfSCTd = pd.DataFrame(SCTd,
                                                                                          columns=list(map(str, TRP)))
                                                                    for i in range(s):
                                                                        for j in range(wpn):
                                                                            L1 = list(map(str, [j]))
                                                                            SCTD[i][j] = dfSCTd.iloc[i][L1[0]]
                                                                    NIS = []
                                                                    NISD = np.zeros(7)
                                                                    for i in range(wpn):
                                                                        for s1 in range(s):
                                                                            c1 = SCTD[s1][i]
                                                                            if c1 < c:
                                                                                NISD[KC[c1][2]] += 1
                                                                        NIS.append(NISD)
                                                                    NISF = [NIS[0] + NIS[1], NIS[2] + NIS[3], NIS[4],
                                                                            NIS[5], NIS[6] + NIS[7], NIS[8] + NIS[9],
                                                                            NIS[13] + NIS[14], NIS[15] + NIS[16],
                                                                            NIS[17] + NIS[18], NIS[19] + NIS[20],
                                                                            NIS[23] + NIS[24], NIS[25] + NIS[26],
                                                                            NIS[27], NIS[28]]
                                                                    FASI = np.arange(0, wpn)
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
                                                                                for i in range(23, 27):
                                                                                    FASI[i] = 49 - i
                                                                            else:
                                                                                FASI[28] = 27
                                                                                FASI[27] = 28
                                                                    SCTDi = SCTD[:]
                                                                    SCTDI = SCTD[:]
                                                                    dfSCTDi = pd.DataFrame(SCTDi,
                                                                                           columns=list(map(str, FASI)))
                                                                    for i in range(s):
                                                                        for j in range(wpn):
                                                                            L2 = list(map(str, [j]))
                                                                            SCTDI[i][j] = dfSCTDi.iloc[i][L2[0]]
                                                                    QTTD = []
                                                                    for a in range(nds):
                                                                        QTD = np.zeros(q)
                                                                        for i in range(q):
                                                                            ttypeq = RTTypeQ[i][:]
                                                                            QU = QU(ttypeq)[:]
                                                                            t1, t2 = QU[Tq[i]]
                                                                            date1 = date(t1)
                                                                            date2 = date(t2)
                                                                            date3 = ADS[a][date1]
                                                                            date4 = ADS[a][date2]
                                                                            qtd = abs(date3 - date4)
                                                                            QTD[i] = min(qtd, 7 - qtd)
                                                                        TTD = np.zeros(t)
                                                                        for i in range(t):
                                                                            ttypet = RTTypeT[i][:]
                                                                            TR = TR(ttypet)[:]
                                                                            t1, t2 = TR[Tt[i]]
                                                                            date1 = date(t1)
                                                                            date2 = date(t2)
                                                                            date3 = ADS[a][date1]
                                                                            date4 = ADS[a][date2]
                                                                            ttd = abs(date3 - date4)
                                                                            TTD[i] = min(ttd, 7 - ttd)
                                                                        QTTD[a] = sum(QTD) + sum(TTD)
                                                                    qttd = QTTD.index(min(QTTD))
                                                                    DF = []
                                                                    bestset = ADS[qttd]
                                                                    for i in range(weekly):
                                                                        for j in range(DPN[bestset[i]]):
                                                                            DF.append(FPD[i] + j)
                                                                    SCTDId = SCTDI[:]
                                                                    SCTDID = SCTDI[:]
                                                                    dfSCTDId = pd.DataFrame(SCTDId,
                                                                                            columns=list(map(str, DF)))
                                                                    for i in range(s):
                                                                        for j in range(wpn):
                                                                            L3 = list(map(str, [j]))
                                                                            SCTDID[i][j] = dfSCTDId.iloc[i][L3[0]]
                                                                    CPTs.append(CPT)
                                                                    std = np.zeros(s)
                                                                    for s1 in range(s):
                                                                        sctd = SCTDID[s1]
                                                                        for t1 in range(wpn):
                                                                            c1 = sctd[t1]
                                                                            c2 = sctd[t1 + 1]
                                                                            std[s1] += DI[c1][c2]
                                                                        for i in range(9):
                                                                            c3 = sctd[IDP[i]]
                                                                            c4 = sctd[IDP[i] - 1]
                                                                            std[s1] -= DI[c3][c4]
                                                                    SAD = (sum(std)) / s
                                                                    rSCT1 = []
                                                                    for s1 in range(s):
                                                                        for i in range(wpn):
                                                                            rSCT1.append(SCTDID[s1][i])
                                                                    rSCT.append(SAD)
                                                                    rSCTs.append(rSCT)
                                                                    ATT.append(rSCT1)
                                                                    stds.append(std)
                                                                    TotSCs.append(TotSC(cSC))
                                                                    """print("checkpoint8")"""
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
                                                            GSTP = RGSTP[-1][:]
                                                            cTTyped = RTTypeP[-1][:]
                                                            CpTpS = RCpTpS[-1][:]
                                                            while gN[4] < g:
                                                                if GSCP[gN[4]] % Prime[P[cN[4]]] == 0:
                                                                    PA1 = PA(cTTyped)
                                                                    gstp = GSTP[gN[4]]
                                                                    z1, z2 = PA1[tN[4]]
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
                                                                    """print("checkpoint6-1")"""
                                                                    i = PIL(FCL, cN[4])
                                                                    PA1 = PA(cTTyped)
                                                                    j1 = CpTpS[i]
                                                                    j2 = MCpTpS[i]
                                                                    j3, j4 = PA1[tN[4]]
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
                                                                        cTTyped = CTType(11, PA1[tN[4]], cTTyped)
                                                                        RTL.append(pa(cTTyped))
                                                                        RTTypeP.append(cTTyped)
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
                                                GSTP = RGSTP[-1][:]
                                                cTTyped = RTTypeT[-1][:]
                                                CpTpS = RCpTpS[-1][:]
                                                while gN[3] < g:
                                                    if GSCP[gN[3]] % Prime[T[cN[3]]] == 0:
                                                        TR1 = TR(cTTyped)
                                                        gstp = GSTP[gN[3]]
                                                        print(tN[3])
                                                        z1, z2 = TR1[tN[3]]
                                                        if NED(gstp, SelIO(Prime, z1, 2)) * (gstp % Prime[z2]) == 0:
                                                            if T[cN[3]] > wc:
                                                                WG = [gN[3]]
                                                                wc = T[cN[3]]
                                                            elif T[cN[3]] == wc:
                                                                WG.append(gN[3])
                                                            gN[3] = 0
                                                            tN[3]+=1
                                                            if tN[3] == RTL:
                                                                gN[3] = g + 1
                                                                back = 1
                                                        else:
                                                            gN[3] += 1
                                                    else:
                                                        gN[3] += 1
                                                    if gN[3] == g:
                                                        """print("checkpoint5-1")"""
                                                        i = PIL(FCL, cN[3])
                                                        TR1 = TR(cTTyped)
                                                        j1 = CpTpS[i]
                                                        j2 = MCpTpS[i]
                                                        j3, j4 = TR1[tN[3]]
                                                        if max(Pm(j1, j3), Pm(j1, j3 + 1), Pm(j1, j4)) == j2:
                                                            gN[3] = 0
                                                            tN[3] +=1
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
                                                            cTTyped = CTType(12, TR1[tN[3]], cTTyped)
                                                            RTL.append(tr(cTTyped))
                                                            RTTypeT.append(cTTyped)
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
                                    GSTP = RGSTP[-1][:]
                                    cTTyped = RTTypeQ[-1][:]
                                    CpTpS = RCpTpS[-1][:]
                                    while gN[2] < g:
                                        if GSCP[gN[2]] % Prime[Q[cN[2]]] == 0:
                                            QU1 = QU(cTTyped)
                                            gstp = GSTP[gN[2]]
                                            z1, z2 = QU1[tN[2]]
                                            if NED(gstp, SelIO(Prime, z1, 2)) * NED(gstp, SelIO(Prime, z2, 2)) == 0:
                                                if Q[cN[2]] > wc:
                                                    WG = [gN[2]]
                                                    wc = Q[cN[2]]
                                                elif Q[cN[2]] == wc:
                                                    WG.append(gN[2])
                                                gN[2] = 0
                                                tN[2]+=1
                                                if tN[2] == RTL[-1]:
                                                    gN[2] = g + 1
                                                    back = 1
                                            else:
                                                gN[2] += 1
                                        else:
                                            gN[2] += 1
                                        if gN[2] == g:
                                            """print("checkpoint4-1")"""
                                            i = PIL(FCL, cN[2])
                                            QU1 = QU(cTTyped)
                                            j1 = CpTpS[i]
                                            j2 = MCpTpS[i]
                                            j3, j4 = QU1[tN[2]]
                                            if max(Pm(j1, j3), Pm(j1, j3 + 1), Pm(j1, j4), Pm(j1, j4 + 1)) == j2:
                                                gN[2] = 0
                                                tN[2]+=1
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
                                                cTTyped = CTType(22, QU1[tN[2]], cTTyped)
                                                RTL.append(qu(cTTyped))
                                                RTTypeQ.append(cTTyped)
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
                        GSTP = RGSTP[-1][:]
                        cTTyped = RTTypeM[-1][:]
                        CpTpS = RCpTpS[-1][:]
                        while gN[1] < g:
                            if GSCP[gN[1]] % Prime[M[cN[1]]] == 0:
                                MO1 = MO(cTTyped)
                                gstp = GSTP[gN[1]]
                                z1 = MO1[tN[1]]
                                if gstp % Prime[z1] == 0:
                                    if M[cN[1]] > wc:
                                        WG = [gN[1]]
                                        wc = M[cN[1]]
                                    elif M[cN[1]] == wc:
                                        WG.append(gN[1])
                                    gN[1] = 0
                                    tN[1]+=1
                                    if tN[1] == RTL[-1]:
                                        gN[1] = g + 1
                                        back = 1
                                else:
                                    gN[1] += 1
                            else:
                                gN[1] += 1
                            if gN[1] == g:
                                """print("checkpoint3-1")"""
                                i = PIL(FCL, cN[1])
                                MO1 = MO(cTTyped)
                                j1 = CpTpS[i]
                                j2 = MCpTpS[i]
                                j3 = MO1[tN[1]]
                                if Pm(j1, j3) == j2:
                                    gN[1] = 0
                                    tN[1]+=1
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
                                    cTTyped = CTType(1, [MO1[tN[1]]], cTTyped)
                                    RTL.append(mo(cTTyped))
                                    RTTypeM.append(cTTyped)
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
            GSTP = RGSTP[-1][:]
            cTTyped = RTTypeD[-1][:]
            CpTpS = RCpTpS[-1][:]
            while gN[0] < g:
                if GSCP[gN[0]] % Prime[D[cN[0]]] == 0:
                    DO1 = DO(cTTyped)
                    gstp = GSTP[gN[0]]
                    z1 = DO1[tN[0]]
                    if NED(gstp, SelIO(Prime, z1, 2)) == 0:
                        if D[cN[0]] > wc:
                            WG = [gN[0]]
                            wc = D[cN[0]]
                        elif D[cN[0]] == wc:
                            WG.append(gN[0])
                        gN[0] = 0
                        tN[0]+=1
                        if tN[0] == RTL[-1]:
                            gN[0] = g + 1
                            back = 1
                    else:
                        gN[0] += 1
                else:
                    gN[0] += 1
                if gN[0] == g:
                    """print("checkpoint2-1")"""
                    i = PIL(FCL, cN[0])
                    DO1 = DO(cTTyped)
                    j1 = CpTpS[i]
                    j2 = MCpTpS[i]
                    j3 = DO1[tN[0]]
                    if max(Pm(j1, j3), Pm(j1, j3 + 1)) == j2:
                        gN[0] = 0
                        tN[0]+=1
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
                        cTTyped = CTType(2, [DO1[tN[0]]], cTTyped)
                        RTL.append(do(cTTyped))
                        RTTypeD.append(cTTyped)
                        cN[0] += 1
                        tN[0] = 0
    if control == 0:
        """print("checkpoint9")"""
        print(9)
        if not list({wc} - set(ECL)):
            LCS2 = ECL[:]
            sub = LCS2.index(wc)
            for ws in range(len(WG)):
                cSC[GS[WG[ws]]][wc] = 0
                cSC[GS[WG[ws]]][FCL[sub]] = 1
                print(GS[WG[ws]], ":", wc, "->", FCL[sub])
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
    """print("checkpoint10")"""
    print(10)
    rSCTDs = []
    dfrSCTds = pd.DataFrame(rSCTs, columns=np.arange(0, wpn * s).append('SAD'))
    lent = len(dfrSCTds.index)
    dfrSCTds['T'] = np.arange(0, lent)
    dfrSCTds.sort_values('SAD', ascending=True)
    TtT1 = dfrSCTds.iloc['T'][:]
    for i in range(lent):
        for j in range(wpn * s):
            L4 = list(map(str, [j]))
            rSCTDs[i][j] = dfrSCTds.iloc[i][L4[0]]
    for i in range(lent):
        OTT = np.array(ATT[TtT1[i]])
        rOTT2 = OTT.reshape(s, wpn)
        rOTT = rOTT2.tolist()
        CPT0 = CPTs[TtT1[i]][:]
        for u in range(12):
            pn = SPC[u]
            le = 40 / pn
            PART = CSP(u)
            for j in range(pn):
                k = PART[j]
                for t1 in range(wpn):
                    if CPT0[k][t1] > 0:
                        RGB2 = CColorSet(PART)[:]
                        plt.fill([t1, t1, t1 + 1, t1 + 1],
                                 [le * (j + 0.25), le * (j + 0.75), le * (j + 0.75), le * (j + 0.25)],
                                 color='#' + '0' * (6 - len(RGB2[j])) + str(RGB2[j]), alpha=0.5)
            plt.title(GT[u])
            plt.show()
        plt.bar(C, TotSCs[TtT1[i]], color='#e35f62')
        plt.xticks(C, C)
        plt.show()
        std2 = stds[TtT1[i]][:]
        s1std = std2[:]
        s2std = std2[:]
        dfs1std = pd.DataFrame(s1std, columns=list(map(str, StS)))
        for j in range(s):
            L5 = list(map(str, [j]))
            s2std[j] = dfs1std.iloc[0][L5[0]]
        plt.rcParams["figure.figsize"] = (12, 12)
        plt.plot(np.arange(s), s2std)
        plt.xlabel('students')
        plt.ylabel('average distance')
        plt.show()
    #
    print("What Time Table?")
    v = int(input())
