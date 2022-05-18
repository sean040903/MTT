import numpy as np
DPN = [6, 7, 4, 6, 6]

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

def PIL(mat, n):
    for i in range(len(mat)):
        if n < mat[i]:
            return i - 1
    return len(mat) - 1


def NED(n, mat):
    for i in range(len(mat)):
        if n % mat[i] == 0:
            return 0
    return 1

daily = max(DPN)
weekly = len(DPN)

FPD = [0]
for i in range(weekly - 1):
    FPD.append(FPD[-1] + DPN[i])


def dates(mat):
    Dates = []
    for i in range(len(mat)):
        Dates.append(PIL(FPD, mat[i]))
    return Dates


def date(n):
    return PIL(FPD, n)


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
TType = FPD
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
print(CAD)
TType.append(0)
print(TType)

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

daily = max(DPN)
weekly = len(DPN)

FPD = [0]
for i in range(weekly - 1):
    FPD.append(FPD[-1] + DPN[i])

def Even(mat):
    eap = []
    for i1 in range(0,weekly):
        if not {i1} - set(CAD):
            if mat[SDPI[CAD.index(i1)]] == 1:
                eap += C2N(FPD[i1], mat[i1] + 2)
        else:
            eap += C2N(FPD[i1], mat[i1] + 2)
    return eap

print(Even(TType))
