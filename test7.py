dfScTd=pd.DataFrame(ScT,columns=list(map(str, CDSD)))
ScTD=dfScTd.values.tolist()
CpTs.append(CpT)
astdw = np.zeros(s)
for s1 in range(s):
    sctd=ScTD[s1]
    for t1 in range(wpn):
        c1=sctd[t1]
        c2=sctd[t1+1]
        astdw[s1]+=DI[c1][c2]
    for i in range(len(UDP)):
        c3=sctd[UDP[i]]
        c4 = sctd[UDP[i]-1]
        astdw[s1] -= DI[c3][c4]
Avgtdw=(sum(astdw))/s
fScT=np.array(ScTD).reshape(-1)
FScT.append(Avgtdw)
FScTs.append(FScT)
RfScT.append(fScT)
RAvgtdw.append(Avgtdw)
RSC.append(np.array(SC).reshape(-1).tolist())