import os
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import scipy.interpolate as sp
import scipy 

def getFilenames():

   
    f1 = 'Q0836-spec_LyaMatrix.tex'
    f2 = 'Q1306-spec_LyaMatrix.tex'
    f3 = 'qso_z_628_lo_res_LyaMatrix.txt'
    f4 = 'qso_z_637_lo_res_LyaMatrix.txt'
    f5 = 'z517.spec_LyaMatrix.tex'
    f6 = 'z521.spec_LyaMatrix.tex'
    f7 = 'z530.spec_LyaMatrix.tex'
    f8 = 'z531.spec_LyaMatrix.tex'
    f9 = 'z541.spec_LyaMatrix.tex'
    f10 = 'z582_hres.spec_LyaMatrix.tex'
    f11 = 'z599_hres.spec_LyaMatrix.tex'
    f = [f1, f2, f3, f4, f5, f6, f8, f9, f10, f11]

    z1 = 5.82
    z2 = 5.99
    z3 = 6.28
    z4 = 6.37
    z5 = 5.17
    z6 = 5.21
    z7 = 5.30
    z8 = 5.31
    z9 = 5.41
    z10 = 5.82
    z11 = 5.99    
    z = [z1, z2, z3, z4, z5, z6, z8, z9, z10, z11]


    return f,z

def getAllFilenames():

    f1 = 'Q0836-spec_LyaMatrix.tex'
    f2 = 'Q1306-spec_LyaMatrix.tex'
    f3 = 'qso_z_628_lo_res_LyaMatrix.txt'
    f4 = 'qso_z_637_lo_res_LyaMatrix.txt'
    f5 = 'z517.spec_LyaMatrix.tex'
    f6 = 'z521.spec_LyaMatrix.tex'
    f7 = 'z530.spec_LyaMatrix.tex'
    f8 = 'z531.spec_LyaMatrix.tex'
    f9 = 'z541.spec_LyaMatrix.tex'
    f10 = 'z574_LyaMatrix.tex'
    f11 = 'z580_LyaMatrix.tex'
    f12 = 'z582_hres.spec_LyaMatrix.tex'
    f13 = 'z582_LyaMatrix.tex'
    f14 = 'z595_LyaMatrix.tex'
    f15 = 'z599_hres.spec_LyaMatrix.tex'
    f16 = 'z599_LyaMatrix.tex'
    f17 = 'z605_LyaMatrix.tex'
    f18 = 'z607_LyaMatrix.tex'
    f19 = 'z614_LyaMatrix.tex'
    f20 = 'z621_LyaMatrix.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f1, f2, f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]
    f = np.hstack((farray1,farray2))
    
    f1 = 'Q0836-spec.tex'
    f2 = 'Q1306-spec.tex'
    f3 = 'qso_z_628_lo_res.txt'
    f4 = 'qso_z_637_lo_res.txt'
    f5 = 'z517.spec.tex'
    f6 = 'z521.spec.tex'
    f7 = 'z530.spec.tex'
    f8 = 'z531.spec.tex'
    f9 = 'z541.spec.tex'
    f10 = 'z574.tex'
    f11 = 'z580.tex'
    f12 = 'z582_hres.spec.tex'
    f13 = 'z582.tex'
    f14 = 'z595.tex'
    f15 = 'z599_hres.spec.tex'
    f16 = 'z599.tex'
    f17 = 'z605.tex'
    f18 = 'z607.tex'
    f19 = 'z614.tex'
    f20 = 'z621.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f1, f2, f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f13, f14, f15, f16, f17, f18, f19, f20]
    fbase = np.hstack((farray1,farray2))


    z1 = 5.82
    z2 = 5.99
    z3 = 6.28
    z4 = 6.37
    z5 = 5.17
    z6 = 5.21
    z7 = 5.30
    z8 = 5.31
    z9 = 5.41
    z10 = 5.74
    z11 = 5.80
    z12 = 5.82
    z13 = 5.82
    z14 = 5.95
    z15 = 5.99
    z16 = 5.99
    z17 = 6.05
    z18 = 6.07
    z19 = 6.14
    z20 = 6.21
    zarray1 = [z1, z2, z3, z4, z5, z6, z8, z9, z10]
    zarray2 = [z11, z12, z13, z14, z15, z16, z17, z18, z19, z20]
    z = np.hstack((zarray1,zarray2))

    return f,fbase,z

def zDistributionOne(mFilename):
    mdata = np.genfromtxt(mFilename)
    zs = np.copy(mdata[0,:])
    return zs

def zDistributionFull():
    f,fbase,z = getAllFilenames()
    nSpectra = len(f)
    z = np.array([])
    for i in range(0,nSpectra):
        if (i != 4):
            ztemp = zDistributionOne(f[i])
        if (i == 0):
            z = ztemp
        z = np.hstack([z,ztemp])

    plt.hist(z,50)
    plt.xlabel('z')
    plt.ylabel('Frequency')
    plt.show(block=False)
    print "you should really be binning specta before doing this..."
    input("Press return to end program")
            
            
def oneStackBinZ(mFilename,z,largeCutoff,smallCutoff,zcut):
    #print "There is a small bug in this code where it is possible that"
    #print "we will neglect dark gaps that overlap with the positive edge"
    #print "of our spectra..."
    superSmallCutoff = 0 #km/s
    testingStack = 0
    #load fluxes and things from matrix
    print mFilename
    mdata = np.genfromtxt(mFilename)
    zs = mdata[0,:]
    flux = np.copy(mdata[1,:])
    sflux = np.copy(mdata[1,:])
    snr = np.copy(mdata[2,:])
    meansnr = np.mean(snr)

    #create variable that is velocity separation 
    c = 3e5
    lambdaA0 = 1216*(1+z)
    lambdas = 1216*(1+zs)
    vs = c*(lambdaA0-lambdas)/lambdaA0

    #create smoothed version of flux. 
    smoothV = 50.0 #km/s
    dv = np.abs(vs[1]-vs[0])
    #print "apparently, resolution is %f km/s..."%(dv)
    smoothn = np.ceil(smoothV/dv)
    #print "smoothing over %f pixels"%(smoothn)
    if (testingStack != 1):
        sflux = smoothSpectra(sflux,smoothn)
    tsnr = snr[:]*np.sqrt(2*smoothn)
    tflux = sflux[:] - 3/tsnr[:]
    sflux[tflux<0] = 0
    
    
    plots = 0
    if plots:
        plt.plot(vs,flux)
        plt.plot(vs,sflux)
        plt.plot(vs,sflux*0)
        plt.plot(vs,3/tsnr)
        plt.show(block=False)
        input('Enter something to continue...')
        plt.close()

    if testingStack:
        flux = createTestFlux(flux)
        sflux = np.copy(flux)
        print np.shape(flux)

    print "beginning stack... prepare for bugs..."
    #ok, so now we actually need to iterate through spectra, stack
    stackCount = 0
    darkGap = 0
    darkIndexReset = -999
    darkIndex = darkIndexReset
    nullValue = -999
    vrange = 1000 #km/s
    nrange = np.ceil(vrange/dv) #number of pixels to stack
    #we'll eventually need to interpolate onto a fixed grid here
    vstack = np.arange(0,2000,dv)
    nrange = np.size(vstack)
    vgrid = np.arange(0,800,2)
    smallCutoff = smallCutoff/dv
    largeCutoff = largeCutoff/dv
    superSmallCutoff = superSmallCutoff/dv
    print smallCutoff,largeCutoff,superSmallCutoff
    smallStackP = nullValue*np.ones(np.shape(vgrid))
    smallStackM = nullValue*np.ones(np.shape(vgrid))
    largeStackP = nullValue*np.ones(np.shape(vgrid))
    largeStackM = nullValue*np.ones(np.shape(vgrid))
    fluxlength = np.size(flux)
    #plt.plot(vs,sflux)
    #plt.xlabel('v (km/s)')
    #plt.ylabel('F(v)')
    for i in range(0,fluxlength):
        if (sflux[i] == 0):
            darkGap = darkGap + 1
            #mark beginning of dark gap
            if (darkIndex == darkIndexReset):
                darkIndex = i
        if ((sflux[i] != 0) & (sflux[i-1]==0)):
            #stack in one direction, check for boundaries too
            if (darkGap >= largeCutoff):
                stackCount = stackCount + 1
                #contributes to large stack
                #need to stack at both edges
                
                #positive stack
                mini = i
                maxi = i + nrange
                plus = nullValue*np.ones(nrange)
                if (maxi >= fluxlength):
                    maxi = (fluxlength - 1)
                nextent = maxi-mini
                if (nextent > 0):
                    condition1 = (zcut>0)&(zs[mini]>=zcut)
                    condition2 = (zcut<0)&(zs[mini]<=np.abs(zcut))
                    if (condition1 | condition2):
                        plus[0:nextent] = flux[mini:maxi]
                        ps = sp.interp1d(vstack,plus)
                        plusgrid = ps(vgrid)
                        largeStackP = np.vstack((largeStackP,plusgrid))
                    #plt.plot(np.array([vs[mini],vs[mini]]),np.array([0,1]))
                    #plt.show(block=False)
                #negative stack
                mini = darkIndex - nrange - 1
                maxi = darkIndex - 1
                minus = nullValue*np.ones(nrange)
                if (mini < 0):
                    mini = 0
                nextent = maxi-mini

                if (nextent > 0):
                    condition1 = (zcut>0)&(zs[maxi]>=zcut)
                    condition2 = (zcut<0)&(zs[maxi]<=np.abs(zcut))
                    if (condition1 | condition2):
                        minus[0:nextent] = flux[mini:maxi]
                        ms = sp.interp1d(vstack,minus)
                        minusgrid = ms(vgrid)
                        largeStackM = np.vstack((largeStackM,minusgrid))
                    #plt.plot(np.array([vs[maxi],vs[maxi]]),np.array([0,1]))
                    #plt.show(block=False)
                    
                
            if ((darkGap < smallCutoff) & (darkGap >= superSmallCutoff)):
                stackCount = stackCount + 1
                #contributes to small stack
                #need to stack at both edges

                #positive stack
                mini = i
                maxi = i + nrange
                plus = nullValue*np.ones(nrange)
                if (maxi >= fluxlength):
                    maxi = (fluxlength - 1)
                nextent = maxi-mini
                if (nextent > 0):
                    condition1 = (zcut>0)&(zs[mini]>=zcut)
                    condition2 = (zcut<0)&(zs[mini]<=np.abs(zcut))
                    if (condition1 | condition2):
                        plus[0:nextent] = flux[mini:maxi]
                        ps = sp.interp1d(vstack,plus)
                        plusgrid = ps(vgrid)
                        smallStackP = np.vstack((smallStackP,plusgrid))
                    #plt.plot(np.array([vs[mini],vs[mini]]),np.array([0,1]),'--')
                    #plt.show(block=False)
                    

                #negative stack
                mini = darkIndex - nrange - 1
                maxi = darkIndex - 1
                minus = nullValue*np.ones(nrange)
                if (mini < 0):
                    mini = 0
                nextent = maxi-mini
                if (nextent > 0):
                    condition1 = (zcut>0)&(zs[maxi]>=zcut)
                    condition2 = (zcut<0)&(zs[maxi]<=np.abs(zcut))
                    if (condition1 | condition2):
                        minus[0:nextent] = flux[mini:maxi]
                        ms = sp.interp1d(vstack,minus)
                        minusgrid = ms(vgrid)
                        smallStackM = np.vstack((smallStackM,minusgrid))
                    #plt.plot(np.array([vs[maxi],vs[maxi]]),np.array([0,1]),'--')
                    #plt.show(block=False)
                    
                    
            darkGap = 0
            darkIndex = darkIndexReset
    #input("Flux and all stacking locations...")
    #plt.close()
    
    print "%s contributed %d stacks..." % (mFilename,stackCount)
    return largeStackP,largeStackM,smallStackP,smallStackM,meansnr


def averageStacks(LP,LM,SP,SM,WLP,WLM,WSP,WSM):
    LM = np.fliplr(LM)
    SM = np.fliplr(SM)
    L = np.vstack((LP,LM))
    S = np.vstack((SP,SM))
    
    #stack weighting matrices like we stack stacked bins...
    WL = np.hstack((WLP,WLM))
    WS = np.hstack((WSP,WSM))

    # WM is a weighting matrix used to weight by snr
    leftRightTest = 0
    if (leftRightTest):
        L = LM
        S = LP
        WL = WLM
        WS = WLP

    leftLeftTest = 0
    if (leftLeftTest):
        L = LP
        S = SP
        WL = WLP
        WS = WSP

    maxflux = 2
    epsilon = 1e-10
    #count how many stacks contribute to each bin
    Lcounts = np.zeros(np.shape(L)[1])+epsilon
    Scounts = np.zeros(np.shape(L)[1])+epsilon
    #average value of each bin
    large = np.zeros(np.shape(L)[1])
    small = np.zeros(np.shape(L)[1])
    largeVar = np.zeros(np.shape(L)[1])
    smallVar = np.zeros(np.shape(L)[1])
    ## Calculate Large Dark Gap Means
    for i in range(0,np.shape(L)[0]):
        for j in range(0,np.shape(L)[1]):
            if (np.abs(L[i,j]) < maxflux):
                Lcounts[j] = Lcounts[j] + WL[i]
                large[j] = large[j] + L[i,j]*WL[i]
    large = large/Lcounts

    ## Calculate Small Dark Gap Means
    for i in range(0,np.shape(S)[0]):
        for j in range(0,np.shape(S)[1]):
            if (np.abs(S[i,j]) < maxflux):
                Scounts[j] = Scounts[j] + WS[i]
                small[j] = small[j] + S[i,j]*WS[i]
    small = small/Scounts
    
    ## Calculate Large Dark Gap Variances
    for i in range(0,np.shape(L)[0]):
        for j in range(0,np.shape(L)[1]):
            if (np.abs(L[i,j]) < maxflux):
                largeVar[j] = largeVar[j] + (L[i,j]-large[j])**2
    largeVar = largeVar/Lcounts**2

    ## Calculate Small Dark Gap Variances
    for i in range(0,np.shape(S)[0]):
        for j in range(0,np.shape(S)[1]):
            if (np.abs(S[i,j]) < maxflux):
                smallVar[j] = smallVar[j] + (S[i,j]-small[j])**2
    smallVar = smallVar/Scounts**2


    return large,small,largeVar,smallVar



def smoothSpectra(fux,n):
    #we want to just smooth this in a natural way and we are 
    #frustrated after looking to Google for help
    fluxcopy = fux[:]
    maxIndex = len(fluxcopy)-n
    for i in range(n.astype(int),maxIndex.astype(int)):
        fluxcopy[i] = np.mean(fluxcopy[i-n:(i+n+1)])
    
    #take care of edges
    fluxcopy[0:n] = np.mean(fluxcopy[0:n])
    fluxcopy[-n::] = np.mean(fluxcopy[-n::])

    return fluxcopy

def createTestFlux(flux):
    flux = np.ones(np.shape(flux))
    middleIndex = np.floor(np.size(flux))/2
    fluxLength = np.size(flux)
    mini = middleIndex - np.floor(fluxLength/6)
    maxi = middleIndex + np.floor(fluxLength/6) + 1
    flux[mini:maxi] = 0
    flux[8:9] = 0
    for i in range(0,np.size(flux)):
        flux[i] = i*flux[i]
    return flux



def fullStack(lmin,lmax,zcut):
    #f,z = getFilenames()
    # This will include files without real noise estimates
    f,fbase,z = getAllFilenames()
    nSpectra = len(f)
    for i in range(0,nSpectra):
        if (i != 4):
            LP,LM,SP,SM,snr = oneStackBinZ(f[i],z[i],lmin,lmax,zcut)

            ### fill in weighting matrix
            print "SNR = %f"%(snr)
            #snr = 1
            fileBase, fileExt = os.path.splitext(fbase[i])
            snrFile ="%s_snrEstimate%s" % (fileBase, fileExt)
            snr = np.genfromtxt(snrFile)
            print "snr = %f" % (snr)
            #print "disabling weighted averaging..."
            WLP = snr*np.ones(np.shape(LP)[0])
            WLM = snr*np.ones(np.shape(LM)[0])
            WSP = snr*np.ones(np.shape(SP)[0])
            WSM = snr*np.ones(np.shape(SM)[0])

            if (i == 0):
                WLPfull = np.copy(WLP)
                WLMfull = np.copy(WLM)
                WSPfull = np.copy(WSP)
                WSMfull = np.copy(WSM)
                
                LPfull = np.copy(LP)
                LMfull = np.copy(LM)
                SPfull = np.copy(SP)
                SMfull = np.copy(SM)
            else:
                if (i != 1):
                    WLPfull = np.hstack((WLPfull,WLP))
                    WLMfull = np.hstack((WLMfull,WLM))
                    WSPfull = np.hstack((WSPfull,WSP))
                    WSMfull = np.hstack((WSMfull,WSM))

                    LPfull = np.vstack((LPfull,LP))
                    LMfull = np.vstack((LMfull,LM))
                    SPfull = np.vstack((SPfull,SP))
                    SMfull = np.vstack((SMfull,SM))
                
            showEachStep = 0
            if (showEachStep):    
                crap1,crap2 = averageStacks(LP,LM,SP,SM)
                plt.plot(crap1)
                plt.show(block=False)
                input("Enter something to continue (This is large stack):")
                plt.plot(crap2,'--')
                plt.show(block=False)
                input("Now we have added the small stack")
    
    large,small,largeVar,smallVar = averageStacks(LPfull,LMfull,SPfull,SMfull,WLPfull,WLMfull,WSPfull,WSMfull)        
    vs = 2*np.arange(0,np.shape(large)[0],1)
    
    plots = 0
    if plots:
        plt.plot(vs,large)
        largeErr = np.sqrt(largeVar)
        #plt.errorbar(vs,large,yerr=largeErr)
        plt.xlabel('v (km/s)')
        plt.show(block=False)
        input("Enter something to continue (This is large stack):")
        plt.plot(vs,small)
        smallErr = np.sqrt(smallVar)
        #plt.errorbar(vs,small,yerr=smallErr)
        plt.legend(('Large Stack','Small Stack'))
        plt.show(block=False)
        input("Now we have added the small stack")
        plt.close()
        plt.plot(vs,largeErr)
        plt.plot(vs,smallErr)
        plt.show(block=False)
        input("these are the errors?")

    return vs,large,small


if __name__ == "__main__":

    lmin = np.array([100,300,500,1000])
    lmax = 300
    zcut = -5
    for i in range(0,np.size(lmin)):
        vs,large,small = fullStack(lmin[i],lmax,zcut)
        if (i == 0):
            plt.plot(vs,small,'--')
        plt.plot(vs,large)
        plt.legend('Small')
        plt.xlabel('v (km/s)')
        plt.ylabel('Stacked Transmission')
        plt.title('Splitting at z = %f'%(zcut))
        legend1 = "L < %d km/s" % (lmax)
        legend2 = "L > %d km/s" % (lmin[0])
        legend3 = "L > %d km/s" % (lmin[1])
        legend4 = "L > %d km/s" % (lmin[2])
        legend5 = "L > %d km/s" % (lmin[3])
        plt.legend((legend1,legend2,legend3,legend4,legend5))
        plt.show(block=False)
    input("Press return to end...")

