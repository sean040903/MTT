import numpy as np
import pandas as pd

# input받는 행렬들#
SC = list(input())
S = list(input())
C = list(input())
FCS = list(input())
LCS = list(input())
KC = list(input())
Prime = list(input())
MCpTpS = list(input())
DI = list(input())

# 기본변수들#
s = len(S)
c = len(C)

# SC가공#
zCC = np.zeros((c, c))
oCC = np.zeros((c, c))
cSC = SC


def CCS(n):
    return list(map(str, np.arange(0, n)))

TCS = []
for s1 in range(s):
    TCS.sum(axis=1)

dfSC = pd.DataFrame(SC, columns=CCS(c))
dfSC['TCS'] = TCS
dfSC['S'] = np.arange(0, s)
dfSC.sort_values('TCS', ascending=True)
dfStS = dfSC['S']
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
        T = +[c1]
        t = +1
    if KC[c1][0] == 2:
        D = +[c1]
        d = +1
    if KC[c1][0] == 11:
        P = +[c1]
        p = +1
    if KC[c1][0] == 1:
        M = +[c1]
        m = +1
    else:
        Q = +[c1]
        q = +1


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


# 요일 분류(병합필요)#
DRP = [[0, 1, 2, 3, 4], [4, 1, 2, 3, 0]]
Mon = [0, 2, 4]
Tu = [6, 8, 10]
We = [13, 15]
Th = [17, 19, 21]
Fr = [24, 26, 28]
FTD = [0, 6, 13, 17, 24, 30]

def date(x):
    return PIL(FTD, x)


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
NDS = [0, 4, 6, 10, 13, 17, 21, 24, 28]

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
        if list(set([b]) - set(ETO)) == []:
            TType[8] = BFN(list(set(NZBF(TType[8])) | set([ETO.index(b)])))
        if list(set([b]) - set(DETO)) == []:
            TType[8] = BFN(list(set(NZBF(TType[8])) | set([DETO.index(b)])))


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
    TR = []
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                TR.append([L1[t1], L2[t2]])
    return TR

def TTType(TType):
    TTType = []
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                TTType.append(CTType(1, L2[t2], ttype))
    return TTType

def tr(TType):
    tr = 0
    L1 = Even(TType)
    don = do(TType)
    for t1 in range(don):
        ttype = CTType(0, L1[t1], TType)
        L2 = Odd(ttype)
        mon = mo(ttype)
        for t2 in range(mon):
            if date(L2[t2]) != date(L1[t1]):
                tr += 1
    return tr

def QU(TType):
    QU = []
    L1 = Even(TType)
    don1 = do(TType)
    for t1 in range(don1):
        ttype = CTType(0, L1[t1], TType)
        L2 = Even(ttype)
        don2 = do(ttype)
        for t2 in range(don2):
            if date(L2[t2]) > date(L1[t1]):
                QU.append([L1[t1], L2[t2]])
    return QU

def QTType(TType):
    QTType = []
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
                QTType.append(qTType)
    return QTType

def qu(TType):
    qu = 0
    L1 = Even(TType)
    don1 = do(TType)
    for t1 in range(don1):
        ttype = CTType(0, L1[t1], TType)
        L2 = Even(ttype)
        don2 = do(ttype)
        for t2 in range(don2):
            if date(L2[t2]) > date(L1[t1]):
                qu += 1
    return qu

def PA(TType):
    PA = []
    L1 = Odd(TType)
    mon1 = mo(TType)
    for t1 in range(mon1):
        ttype = CTType(1, L1[t1], TType)
        L2 = Odd(ttype)
        mon2 = mo(ttype)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                PA.append([L1[t1], L2[t2]])
    return PA

def PTType(TType):
    PTType = []
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
                PTType.append(pTType)
    return PTType

def pa(TType):
    pa = 0
    L1 = Odd(TType)
    mon1 = mo(TType)
    for t1 in range(mon1):
        ttype = CTType(1, L1[t1], TType)
        L2 = Odd(ttype)
        mon2 = mo(ttype)
        for t2 in range(mon2):
            if date(L2[t2]) > date(L1[t1]):
                pa += 1
    return pa


# 색(나중에)#


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


# main code#
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
    GSC = GS = []
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

    # D코드#
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
                                                                    CPT[k1][k2] = CPT[k1][k2 + 1] = CPT[k1][k3] = CPT[k1][k3 + 1] = k4
                                                                    CkT[k1][k2] = CkT[k1][k2 + 1] = CkT[k1][k3] = CkT[k1][k3 + 1] = k5
                                                                    CCT[k1][k2] = CCT[k1][k2 + 1] = CCT[k1][k3] = CCT[k1][k3 + 1] = k1

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

                                                                SPT = cSC * CPT
                                                                SkT = cSC * CkT
                                                                SCT = cSC * CCT

                                                                if SkDc(SkT, SCT) == 0:
                                                                    AT = thirtyTTRP(Mon, Tu, We, Th, Fr, SCT)
                                                                    SCTD = SCTd = SCT
                                                                    TRP = np.zeros(30)
                                                                    for i in range(30):
                                                                        TRP[AT[i]] = i
                                                                    dfSCTd = pd.DataFrame(SCTd,columns=list(map(str, TRP)))
                                                                    for i in range(s):
                                                                        for j in range(30):
                                                                            L1 = list(map(str, [j]))
                                                                            SCTD[i][j] = dfSCTd.ix[i][L1[0]]


                                                                    # QDTF code#
                                                                    NIS = []
                                                                    NISD = np.zeros(7)
                                                                    for i in range(30):
                                                                        for s1 in range(s):
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
                                                                    dfSCTDi = pd.DataFrame(SCTDi,columns=list(map(str, FASI)))
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
                                                                    dfSCTDId = pd.DataFrame(SCTDId,columns=list(map(str, DF)))
                                                                    for i in range(s):
                                                                        for j in range(30):
                                                                            L3 = list(map(str, [j]))
                                                                            SCTDID[i][j] = dfSCTDId.ix[i][L3[0]]

                                                                    # STD code#
                                                                    CPTs.append(CPT)
                                                                    std=np.zeros(s)
                                                                    for s1 in range(s):
                                                                        sctd=SCTDID[s1]
                                                                        for t1 in range(30):
                                                                            c1=sctd[t1]
                                                                            c2=sctd[t1+1]
                                                                            std[s1]+=DI[c1][c2]
                                                                        for i in range(9):
                                                                            c3=sctd[NDS[i]]
                                                                            c4=sctd[NDS[i]-1]
                                                                            sctd[s1]-=DI[c3][c4]
                                                                    SAD=(sum(std))/s
                                                                    rSCT1=[]
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
                                                                    if Cp == []:
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
                                                                    [j3, j4] = PA[tN[4]]
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
                                                        if Ct == []:
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
                                            if Cq == []:
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
                                                    back
                                            else:
                                                gN[2] += 1
                                        else:
                                            gN[2] += 1
                                        if gN[2] == g:
                                            i = PIL(FCS, cN[2])
                                            QU1 = QU(TType)
                                            j1 = CpTpS[i]
                                            j2 = MCpTpS[i]
                                            [j3, j4] == QU1[tN[2]]
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
                                                for j1 in range(g):
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
                                if Cm == []:
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
                    if Cd == []:
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
        if list(set([wc]) - set(LCS)) == []:
            sub = LCS.index(wc)
            for ws in range(len(WG)):
                cSC[GS[WG[ws]]][wc] = 0
                cSC[GS[WG[ws]]][FCS(sub)] = 1
                print(GS[WG[ws]], ":", wc, "->", FCS[sub])
        else:
            for sub2 in range(len(WG)):
                cSC[GS[WG[ws]]][wc] = 0
                cSC[GS[WG[ws]]][wc + 1] = 1
                print(GS[WG[ws]], ":", wc, "->", wc + 1)
        if dfSC == pd.DataFrame(cSC, columns=CCS(c)):
            if rSCTs == []:
                print("ERROR")
            else:
                keep = 1
        else:
            control = 1
            print("control code deactivated")


if keep==1:

