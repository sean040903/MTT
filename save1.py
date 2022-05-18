def Even(TType):
    list1 = EPs(np.arange(0, PIL(FPD,TType[0])))
    list2 = EPs(np.arange(3, TType[1] + 1))
    list3 = EPs(np.arange(6, TType[2] + 1))
    list4 = EPs(np.arange(8, TType[3] + 1)) * TType[5]
    list5 = EPs(np.arange(11, TType[4] + 1)) * TType[6]
    list6= list(set(list1) | set(list2) | set(list3) | set(list4) | set(list5))
    return np.array(list6)


def Odd(TType):
    list1 = EPs(np.arange(0, TType[0] + 1))
    list2 = EPs(np.arange(3, TType[1] + 1))
    list3 = EPs(np.arange(6, TType[2] + 1))
    list4 = EPs(np.arange(8, TType[3] + 1)) * TType[5]
    list5 = EPs(np.arange(11, TType[4] + 1)) * TType[6]
    list6 = VEPs(NZBF(TType[8]))
    list7 = [23 * TType[7], 12]
    list8=list(set(list1) | set(list2) | set(list3) | set(list4) | set(list5) | set(list6) | set(list7))
    return np.array(list8)