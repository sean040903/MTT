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

