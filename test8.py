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