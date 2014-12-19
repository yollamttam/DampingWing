import numpy as np
import matplotlib.pyplot as plt


def CompareAll():
    rall = np.genfromtxt('hst_comp01_all.asc')
    rq = np.genfromtxt('hst_comp01_rq.asc')
    rl = np.genfromtxt('hst_comp01_rl.asc')
    original = np.genfromtxt('LBQS.lis')

    lambdaAll = rall[:,0]
    lambdaRQ = rq[:,0]
    lambdaRL = rl[:,0]
    lambdaO = original[:,0]

    fluxAll = rall[:,1]
    fluxRQ = rq[:,1]
    fluxRL = rl[:,1]
    fluxO = original[:,1]

    # arbitrarily normalize "original" template to the All template at 
    # 1100 Angstroms
    normLambda = 1100
    iAll = np.size(lambdaAll[lambdaAll<=normLambda])-1
    normAll = fluxAll[iAll]
    iO = np.size(lambdaO[lambdaO<=normLambda])-1
    normO = fluxO[iO]
    fluxO = fluxO*normAll/normO

    Lya = 1216
    Lyb = 1026
    Oxy = 1034

    lambdaBlue = 1000
    lambdaRed = 1300
    maxA = np.max(fluxAll[(lambdaAll>lambdaBlue)&(lambdaAll<lambdaRed)])
    maxRQ = np.max(fluxRQ[(lambdaRQ>lambdaBlue)&(lambdaRQ<lambdaRed)])
    maxRL = np.max(fluxRL[(lambdaRL>lambdaBlue)&(lambdaRL<lambdaRed)])
    maxO = np.max(fluxO[(lambdaO>lambdaBlue)&(lambdaO<lambdaRed)])
    maxFlux = np.max(np.array([maxA,maxRQ,maxRL,maxO]))

    plt.plot(lambdaAll,fluxAll)
    plt.plot(lambdaRQ,fluxRQ)
    plt.plot(lambdaRL,fluxRL)
    plt.plot(lambdaO,fluxO)
    plt.plot([Lya,Lya],[0,maxFlux])
    plt.plot([Lyb,Lyb],[0,maxFlux])
    plt.plot([Oxy,Oxy],[0,maxFlux])
    plt.xlabel('Wavelength (A)')
    plt.ylabel('Template Flux')
    plt.axis([lambdaBlue,lambdaRed,0,maxFlux])
    plt.legend(('HST All','HST Radio Quiet','HST Radio Loud','Original Template','Lya','Lyb','OVI'))
    plt.show(block=False)
    input("Press return to end program...")








if __name__ == "__main__":

    CompareAll()
