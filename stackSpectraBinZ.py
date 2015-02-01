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

    # comment out duplicate spectra
    # f1 = 'Q0836-spec_LyaMatrix.tex'
    # f2 = 'Q1306-spec_LyaMatrix.tex'
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
    # f13 = 'z582_LyaMatrix.tex'
    f14 = 'z595_LyaMatrix.tex'
    f15 = 'z599_hres.spec_LyaMatrix.tex'
    f16 = 'z599_LyaMatrix.tex'
    f17 = 'z605_LyaMatrix.tex'
    f18 = 'z607_LyaMatrix.tex'
    f19 = 'z614_LyaMatrix.tex'
    f20 = 'z621_LyaMatrix.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f14, f15, f16, f17, f18, f19, f20]
    f = np.hstack((farray1,farray2))
    
    # f1 = 'Q0836-spec.tex'
    # f2 = 'Q1306-spec.tex'
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
    # f13 = 'z582.tex'
    f14 = 'z595.tex'
    f15 = 'z599_hres.spec.tex'
    f16 = 'z599.tex'
    f17 = 'z605.tex'
    f18 = 'z607.tex'
    f19 = 'z614.tex'
    f20 = 'z621.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f14, f15, f16, f17, f18, f19, f20]
    fbase = np.hstack((farray1,farray2))


    # z1 = 5.82
    # z2 = 5.99
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
    # z13 = 5.82
    z14 = 5.95
    z15 = 5.99
    z16 = 5.99
    z17 = 6.05
    z18 = 6.07
    z19 = 6.14
    z20 = 6.21
    zarray1 = [z3, z4, z5, z6, z8, z9, z10]
    zarray2 = [z11, z12, z14, z15, z16, z17, z18, z19, z20]
    z = np.hstack((zarray1,zarray2))

    return f,fbase,z

def getAllFilenamesLyb():

    # f1 = 'Q0836-spec_LybMatrix.tex'
    # f2 = 'Q1306-spec_LybMatrix.tex'
    f3 = 'qso_z_628_lo_res_LybMatrix.txt'
    f4 = 'qso_z_637_lo_res_LybMatrix.txt'
    f5 = 'z517.spec_LybMatrix.tex'
    f6 = 'z521.spec_LybMatrix.tex'
    f7 = 'z530.spec_LybMatrix.tex'
    f8 = 'z531.spec_LybMatrix.tex'
    f9 = 'z541.spec_LybMatrix.tex'
    f10 = 'z574_LybMatrix.tex'
    f11 = 'z580_LybMatrix.tex'
    f12 = 'z582_hres.spec_LybMatrix.tex'
    # f13 = 'z582_LybMatrix.tex'
    f14 = 'z595_LybMatrix.tex'
    f15 = 'z599_hres.spec_LybMatrix.tex'
    f16 = 'z599_LybMatrix.tex'
    f17 = 'z605_LybMatrix.tex'
    f18 = 'z607_LybMatrix.tex'
    f19 = 'z614_LybMatrix.tex'
    f20 = 'z621_LybMatrix.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f14, f15, f16, f17, f18, f19, f20]
    f = np.hstack((farray1,farray2))
    
    # f1 = 'Q0836-spec.tex'
    # f2 = 'Q1306-spec.tex'
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
    # f13 = 'z582.tex'
    f14 = 'z595.tex'
    f15 = 'z599_hres.spec.tex'
    f16 = 'z599.tex'
    f17 = 'z605.tex'
    f18 = 'z607.tex'
    f19 = 'z614.tex'
    f20 = 'z621.tex'
    
    print "We are skipping %s since its fucked..."%(f7)
    farray1 = [f3, f4, f5, f6, f8, f9, f10]
    farray2 = [f11, f12, f14, f15, f16, f17, f18, f19, f20]
    fbase = np.hstack((farray1,farray2))


    # z1 = 5.82
    # z2 = 5.99
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
    #z13 = 5.82
    z14 = 5.95
    z15 = 5.99
    z16 = 5.99
    z17 = 6.05
    z18 = 6.07
    z19 = 6.14
    z20 = 6.21
    zarray1 = [z3, z4, z5, z6, z8, z9, z10]
    zarray2 = [z11, z12, z14, z15, z16, z17, z18, z19, z20]
    z = np.hstack((zarray1,zarray2))

    return f,fbase,z

def findWeightings():
    f,fbase,z = getAllFilenames()
    # common res means we are stacking spectra that have been smoothed to a common resolution and sampled at the same velocity spacing. We want to skip over the parts of the code that account for spectra having different resoluations. We have also already calculated the estimated noise variance per smoothed pixel
    commonRes = 1
    nspectra = len(f)
    for i in range(0,nspectra):
        if commonRes:
            fileBase, fileExt = os.path.splitext(f[i])
            addString = "smoothed"
            print f[i],fileBase,fileExt,addString
            infile = "%s_%s.tex" % (fileBase,addString)
        else:
            infile = f[i]
        a,b,c = findOneWeighting(z[i],infile,fbase[i],commonRes)
        plt.plot(i,a,'x')
        plt.plot(i,b,'o')
        plt.plot(i,1/c,'v')
        
    plt.legend(['Data Noise Sigma','Estimated Noise','1/Weighting'])
    plt.show(block=False)
    input("Press enter to end program...")

def findOneWeighting(z,filename,basefile,commonRes):
    # This should find the individual weightings for each spectra
    # This will use sigmaF, smoothed on FWHM = 140 km/s
    # alphabar, ratio of unsmoothed sigmaN_blue^2/sigmaN_red^2
    # omega_i, ratio of sigmaN_smoothed,i^2/sigmaN_i^2

    print "calculating weighting for %s..." % (filename)

    sigmaF = 0.161197
    alphabar = 0.087133
    c = 3e5

    fileBase, fileExt = os.path.splitext(basefile)
    estimateFile = "%s_snrEstimate%s" % (fileBase,fileExt)
    redSNR = np.genfromtxt(estimateFile)
    fileBase, fileExt = os.path.splitext(filename)
    blueSmoothedFile = "%s_NratioFromSmoothing.tex" % (fileBase)
    omega = np.genfromtxt(blueSmoothedFile)

    data = np.genfromtxt(filename,delimiter=',')
    print np.shape(data)
    datasnr = np.mean(data[2,:])
    z1 = data[0,1]
    z2 = data[0,2]
    lambdaA0 = 1216*(1+z)
    lambdaA1 = 1216*(1+z1)
    lambdaA2 = 1216*(1+z2)
    v1 = np.abs(c*(lambdaA1-lambdaA0)/lambdaA0)
    v2 = np.abs(c*(lambdaA2-lambdaA0)/lambdaA0)
    dv = np.abs(v1-v2)
    N = np.ceil(140/dv)
    if (abs(N-1) > .1):
        print "maybe something wrong with re-sampling..."
        print "unless you're no longer using common res spectra..."
    datasnr = datasnr*np.sqrt(N)
    # important quantities: redSNR, omega, alphabar, sigmaF
    sigmaNred = 1/redSNR
    varNred = sigmaNred**2
    varNblue = alphabar*varNred
    varNblueSmoothed = omega*varNblue

    if commonRes:
        datasnr = np.mean(data[2,:])
        varNblueSmoothed = 1/datasnr**2

    # this is the actual weighting
    weighting = 1/(varNblueSmoothed+sigmaF**2)
    
    # lets save it to a new file
    fileBase, fileExt = os.path.splitext(filename)
    weightFile = "%s_weighting%s" % (fileBase,fileExt)
    outfile = open(weightFile,'w')
    outfile.write("%f"%(weighting))
    outfile.close()

    return 1/datasnr**2,varNblueSmoothed,weighting

    print "saved weighting, moving on..."


def findMeanF(zarray,counts,fluxes,pcounts,pfluxes,mFilename,z):

    mdata = np.genfromtxt(mFilename)
    zs = mdata[0,:]
    flux = np.copy(mdata[1,:])
    snr = np.copy(mdata[2,:])

    #create variable that is velocity separation 
    c = 3e5
    lambdaA0 = 1216*(1+z)
    lambdas = 1216*(1+zs)
    vs = c*(lambdaA0-lambdas)/lambdaA0

    #so we'd like to know the flux including and not including
    #large dark gaps. we'll need to smooth spectra to find dark 
    #gaps and then find mean outside of those
    smoothV = 50.0
    dv = np.abs(vs[1]-vs[0])
    smoothn = np.ceil(smoothV/dv)
    sflux = np.copy(flux)
    sflux = smoothSpectra(sflux,smoothn)
    tsnr = snr[:]*np.sqrt(2*smoothn)
    t = np.mean(3/tsnr)

    dz = np.abs(zarray[1]-zarray[0])
    minz = np.min(zs)
    maxz = np.max(zs)
    nz = np.size(zarray)
    for i in range(0,nz):
        if ((minz < zarray[i])&(maxz>zarray[i])):
            # fluxrange = flux[(zs>zarray[i])&(zs<(zarray[i]+dz))]
            # sfluxrange = sflux[(zs>zarray[i])&(zs<(zarray[i]+dz))]
            # pfluxrange = fluxrange[sfluxrange>t]
            fluxrange = flux[zs>zarray[i]]
            sfluxrange = sflux[zs>zarray[i]]
            pfluxrange = fluxrange[sfluxrange>t]
            meanf = np.mean(fluxrange)
            

            fluxes[i] = fluxes[i] + meanf
            counts[i] = counts[i] + 1
            
            if (np.size(pfluxrange>0)):
                print "THIS IS NONSENSICAL AND PROBABLY BROKEN!!!"
                meanpf = np.mean(pfluxrange)
                pfluxes[i] = pfluxes[i] + meanpf
                pcounts[i] = pcounts[i] + 1
                
    return fluxes,counts,pfluxes,pcounts


def FvsZ():
    f,fbase,z = getAllFilenames()
    nSpectra = len(f)
    
    zmin = 4.36248913251 
    zmax = 6.21088364792
    dz = .05
    epsilon = 1e-10
    zarray = np.arange(zmin,zmax+dz,dz)
    counts = np.zeros(np.shape(zarray))+epsilon
    fluxes = np.zeros(np.shape(zarray))
    pcounts = np.zeros(np.shape(zarray))+epsilon
    pfluxes = np.zeros(np.shape(zarray))
    for i in range(0,nSpectra):
        if (i != 4):
            fluxes,counts,pfluxes,pcounts = findMeanF(zarray,counts,fluxes,pcounts,pfluxes,f[i],z[i])
            
    fluxes = fluxes/counts
    pfluxes = pfluxes/pcounts
    zarray = zarray + dz/2
    plt.plot(zarray[fluxes>0],fluxes[fluxes>0])
    plt.plot(zarray[pfluxes>0],pfluxes[pfluxes>0])
    plt.xlabel('$z_{min}$')
    plt.ylabel('$<F|z>z_{min}>$')
    plt.legend(['<F>','<F|flux>'])
    plt.title('Mean Transmission vs. Z (First Pass)')
    plt.show(block=False)
    input("Press Enter to end program...")

def FvsZ_ExcludeLs():
    f,fbase,z = getAllFilenames()
    nSpectra = len(f)
    zcut = 1
    Lminmin = 10
    Lminmax = 1000
    dLmin = 10
    epsilon = 1e-10
    Lminarray = np.arange(Lminmin,Lminmax+dLmin,dLmin)
    counts = np.zeros(np.shape(Lminarray))+epsilon
    pcounts = np.zeros(np.shape(Lminarray))+epsilon
    tcounts = np.zeros(np.shape(Lminarray))+epsilon
    fluxes = np.zeros(np.shape(Lminarray))
    pfluxes = np.zeros(np.shape(Lminarray))
    tfluxes = np.zeros(np.shape(Lminarray))
    for i in range(0,nSpectra):
        if (i != 4):
            fluxes,counts,pfluxes,pcounts,tfluxes,tcounts = findMeanF_ExcludeLs(Lminarray,counts,fluxes,pcounts,pfluxes,tcounts,tfluxes,f[i],z[i],zcut)
            
    fluxes = fluxes/counts
    pfluxes = pfluxes/pcounts
    tfluxes = tfluxes/tcounts
    plt.plot(Lminarray,fluxes)
    plt.plot(Lminarray,pfluxes)
    plt.plot(Lminarray,tfluxes)
    plt.xlabel('$L_{min}$')
    plt.ylabel('$<F|L<L_{min}>$')
    plt.legend(['<F|L<L_{min}>','<F|flux>','<F>'])
    plt.axis([0,1000,0.05,0.31])
    plt.title('Mean Transmission Outside of Dark Gaps with L > Lmin')
    plt.show(block=False)
    input("Press Enter to end program...")

def zDistributionOne(mFilename):
    mdata = np.genfromtxt(mFilename)
    zs = np.copy(mdata[0,:])

    #we need to put this on some kind of uniform grid
    minz = np.min(zs)
    maxz = np.max(zs)
    dz = .001
    newzs = np.arange(minz,maxz+dz,dz)

    
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
    plt.title('Z Histogram without Binning Spectra')
    plt.show(block=False)
    
    print np.min(z),np.max(z)

    input("Press return to end program")
            
def tauDistributionOne(mFilename,taupoints):
    
    mdata = np.genfromtxt(mFilename)
    zs = np.copy(mdata[0,:])
    flux = np.copy(mdata[1,:])
    snr = np.copy(mdata[2,:])
    #we need to put this on some kind of uniform grid
    minz = np.min(zs)
    maxz = np.max(zs)
    dz = .15
    newzs = np.arange(minz,maxz+dz,dz)
    
    #you're going to have to do something like this
    #in order to account for F < 2\sigma_N
    #tsnr = snr[:]*np.sqrt(2*smoothn)

    for i in range(0,np.size(newzs)-1):
        zmin = newzs[i]
        zmax = newzs[i+1]
        
        #mean redshift and flux of this bin
        zpoint = np.mean(zs[(zs>=zmin)&(zs<=zmax)])
        fpoint = np.mean(flux[(zs>=zmin)&(zs<=zmax)])
        
        #mean signal-to-noise, then noise sigma, accounting for several bins
        meansnr = np.mean(snr[(zs>=zmin)&(zs<=zmax)])
        npixels = np.size(snr[(zs>=zmin)&(zs<=zmax)])
        nsigma = 1/(meansnr*np.sqrt(npixels))
        
        #flux below 2sigman gets set to 2sigman.
        lowerBound = 0
        cutoff = 2*nsigma
        if (fpoint <= 0):
            fpoint = cutoff
            lowerBound = 1
        
        cutoffTau = -1*np.log(cutoff)
        if (cutoffTau > 4):
            print cutoffTau
        


        taueff = -1*np.log(fpoint)
        datum = np.array([zpoint,taueff,lowerBound])
        taupoints = np.vstack((taupoints,datum))
            

    return taupoints


def tauScatterPlot():
    f,fbase,z = getAllFilenames()
    nSpectra = len(f)
    taupoints = np.array([0,0,99])
    for i in range(0,nSpectra):
        taupoints = tauDistributionOne(f[i],taupoints)

    
    print np.shape(taupoints)
    z = taupoints[:,0]
    taus = taupoints[:,1]
    lowerBound = taupoints[:,2]
    fitz = np.arange(np.min(z[z>0]),np.max(z[z>0]),.01)
    fitz = np.arange(3,6.4,.01)
    plt.plot(z[lowerBound==0],taus[lowerBound==0],'x')
    plt.plot(fitz,.85*((1+fitz)/5)**(4.3),'--')
    plt.plot(z[lowerBound==1],taus[lowerBound==1],'o')
    plt.xlabel('z')
    plt.ylabel('Effective Optical Depth')
    #plt.title('Effective Optical Depth vs. z')
    plt.axis([3,6.4,0,8])
    plt.show(block=False)

    input("Press return to end program")

            
def oneStackBinZ(mFilename,z,largeCutoff,smallCutoff,zcut,t,Larray):
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
    tflux = sflux[:] - t/tsnr[:]
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
    vgrid = np.arange(0,1800,2)
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
    #plt.ylabel('F_{unsmoothed}(v)')
    for i in range(0,fluxlength):
        if (sflux[i] == 0):
            darkGap = darkGap + 1
            #mark beginning of dark gap
            if (darkIndex == darkIndexReset):
                darkIndex = i
        if ((sflux[i] != 0) & (sflux[i-1]==0) & (i > 0)):
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
                        Ldark = dv*np.abs(darkIndex-i)
                        Lt = t/snr[i]
                        addPoint = np.array([Lt,Ldark])
                        Larray = np.vstack((Larray,addPoint))
                        
                        # plt.plot(np.array([vs[mini],vs[mini]]),np.array([0,1]))
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
                        minus[-1*nextent::] = flux[mini:maxi]
                        ms = sp.interp1d(vstack,minus)
                        minusgrid = ms(vgrid)
                        largeStackM = np.vstack((largeStackM,minusgrid))
                        Ldark = dv*nextent
                        Lt = t/snr[i]
                        if (minus[-2] == nullValue):
                            print "ERRORRR!!!!"
                            input('yep...')
                        if (minusgrid[-1] < -10):
                            input('an error here...')           
                
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
                        Ldark = dv*np.abs(darkIndex-i)
                        Lt = t/snr[i]
                        addPoint = np.array([Lt,Ldark])
                        Larray = np.vstack((Larray,addPoint))
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
                        minus[-1*nextent::] = flux[mini:maxi]
                        ms = sp.interp1d(vstack,minus)
                        minusgrid = ms(vgrid)
                        smallStackM = np.vstack((smallStackM,minusgrid))
                        Ldark = dv*nextent
                        Lt = t/snr[i]
                        if (minus[-1] == nullValue):
                            print "ERRORRR!!!!"
                            input('yep...')
                        if (ms[-1] < -10):
                            input('an error here...')
                        # plt.plot(np.array([vs[maxi],vs[maxi]]),np.array([0,1]),'--')
                    #plt.show(block=False)
                    
                    
            darkGap = 0
            darkIndex = darkIndexReset

    #plt.axis([np.min(vs),np.max(vs),-0.1,1.2])
    #plt.show(block=False)
    #input("Flux and all stacking locations...")
    #plt.close()
    
    print "%s contributed %d stacks..." % (mFilename,stackCount)
    return largeStackP,largeStackM,smallStackP,smallStackM,meansnr,Larray


def findMeanF_ExcludeLs(Lminarray,counts,fluxes,pcounts,pfluxes,tcounts,tfluxes,mFilenameA,z,zcut):
    print "this will not work if you are using Lyb..."
    mFilenameB = mFilenameA
    # print "There is a small bug in this code where it is possible that"
    # print "we will neglect dark gaps that overlap with the positive edge"
    # print "of our spectra..."
    superSmallCutoff = 0 #km/s
    largeCutoff = np.min(Lminarray)
    t = 3
    # load Lya fluxes and things from matrix
    print mFilenameA,mFilenameB
    mdataA = np.genfromtxt(mFilenameA)
    zsA = mdataA[0,:]
    fluxA = np.copy(mdataA[1,:])
    sfluxA = np.copy(mdataA[1,:])
    snrA = np.copy(mdataA[2,:])
    meansnrA = np.mean(snrA)

    # load Lyb information
    mdataB = np.genfromtxt(mFilenameB)
    zsB = mdataB[0,:]
    fluxB = np.copy(mdataB[1,:])
    sfluxB = np.copy(mdataB[1,:])
    snrB = np.copy(mdataB[2,:])
    meansnrB = np.mean(snrB)

    #create variable that is velocity separation 
    c = 3e5
    lambdaA0 = 1216*(1+z)
    lambdasA = 1216*(1+zsA)
    lambdaB0 = 1026*(1+z)
    lambdasB = 1026*(1+zsB)
    vsA = c*(lambdaA0-lambdasA)/lambdaA0
    vsB = c*(lambdaB0-lambdasB)/lambdaB0
    
    #create smoothed version of flux. 
    smoothV = 50.0 #km/s
    dvB = np.abs(vsB[1]-vsB[0])
    dvA = np.abs(vsA[1]-vsA[0])
    #print "apparently, resolution is %f km/s..."%(dv)
    smoothn = np.ceil(smoothV/dvB)
    #print "smoothing over %f pixels"%(smoothn)
    
    sfluxB = smoothSpectra(sfluxB,smoothn)
    tsnr = snrB[:]*np.sqrt(2*smoothn)
    tflux = sfluxB[:] - t/tsnr[:]
    sfluxB[tflux<0] = 0

    # ok, so now we actually need to iterate through spectra, stack
    darkGap = 0
    darkIndexReset = -999
    darkIndex = darkIndexReset
    nullValue = -999
    vrange = 1000 #km/s
    nrange = np.ceil(vrange/dvA) #number of pixels to stack
    #we'll eventually need to interpolate onto a fixed grid here
    vstack = np.arange(0,2000,dvA)
    nrange = np.size(vstack)
    nrangeB = np.round(nrange*dvA/dvB)
    vgrid = np.arange(0,800,2)
    largeCutoff = largeCutoff/dvB
    superSmallCutoff = superSmallCutoff/dvB
    fluxlength = np.size(fluxB)
    #plt.plot(vs,sflux)
    #plt.xlabel('v (km/s)')
    #plt.ylabel('F_{unsmoothed}(v)')

    zminA = np.min(zsA)
    zmaxA = np.max(zsA)
    
    # for each pixel in the flux, we record the length of the dark gap that it is a part of
    trackL = np.zeros(np.shape(fluxA))
    
    for i in range(0,fluxlength):
        if (sfluxB[i] == 0):
            darkGap = darkGap + 1
            #mark beginning of dark gap
            if (darkIndex == darkIndexReset):
                darkIndex = i
        if ((sfluxB[i] != 0) & (sfluxB[i-1]==0) & (i > 0)):
            #stack in one direction, check for boundaries too
            # if (darkGap >= largeCutoff):
            Ldark = dvB*np.abs(darkIndex-i)
            trackL[darkIndex:i] = Ldark
                
                    
            darkGap = 0
            darkIndex = darkIndexReset
    
    maxGap = np.max(trackL)
    for i in range(0,np.shape(Lminarray)[0]):
        Lmin = Lminarray[i]
        if (maxGap >= Lmin):
            tempFlux = np.mean(fluxA[trackL<Lmin])
            fluxes[i] = fluxes[i] + tempFlux
            counts[i] = counts[i] + 1
            tempPflux = np.mean(fluxA[trackL==0])
            pfluxes[i] = pfluxes[i] + tempPflux
            pcounts[i] = pcounts[i] + 1
            tempTflux = np.mean(fluxA)
            tfluxes[i] = tfluxes[i] + tempTflux
            tcounts[i] = tcounts[i] + 1
            
    return fluxes,counts,pfluxes,pcounts,tfluxes,tcounts


def oneStackBinZLyb(mFilenameA,mFilenameB,z,largeCutoff,smallCutoff,zcut,t,Larray):
    #print "There is a small bug in this code where it is possible that"
    #print "we will neglect dark gaps that overlap with the positive edge"
    #print "of our spectra..."
    superSmallCutoff = 0 #km/s
    testingStack = 0
    requireLyab = 0
    # would you like plots of stacking locations on top of spectra?
    stackPlot = 0
    print mFilenameA,mFilenameB
    # load Lya fluxes and things from matrix
    mdataA = np.genfromtxt(mFilenameA)
    if (np.shape(np.shape(mdataA))==(1,)):
        mdataA = np.genfromtxt(mFilenameA,delimiter=',')
    zsA = mdataA[0,:]
    fluxA = np.copy(mdataA[1,:])
    sfluxA = np.copy(mdataA[1,:])
    snrA = np.copy(mdataA[2,:])
    meansnrA = np.mean(snrA)

    # load Lyb information
    mdataB = np.genfromtxt(mFilenameB)
    if (np.shape(np.shape(mdataB))==(1,)):
        mdataB = np.genfromtxt(mFilenameB,delimiter=',')
    zsB = mdataB[0,:]
    fluxB = np.copy(mdataB[1,:])
    sfluxB = np.copy(mdataB[1,:])
    snrB = np.copy(mdataB[2,:])
    meansnrB = np.mean(snrB)

    # create variable that is velocity separation 
    c = 3e5
    lambdaA0 = 1216*(1+z)
    lambdasA = 1216*(1+zsA)
    lambdaB0 = 1026*(1+z)
    lambdasB = 1026*(1+zsB)
    vsA = c*(lambdaA0-lambdasA)/lambdaA0
    vsB = c*(lambdaB0-lambdasB)/lambdaB0
    dvB = np.abs(vsB[1]-vsB[0])
    dvA = np.abs(vsA[1]-vsA[0])
    

    # create smoothed version of flux. 
    # threshold smoothed flux based on dark-pixel criteria
    smoothV = 50.0 #km/s
    # Lya
    smoothnA = np.ceil(smoothV/dvA)    
    tsnrA = np.copy(snrA)*np.sqrt(2*smoothnA)
    tfluxA = np.copy(sfluxA) - t/np.copy(tsnrA)
    sfluxA = smoothSpectra(sfluxA,smoothnA)
    sfluxA[tfluxA<0] = 0
    # Lyb
    smoothnB = np.ceil(smoothV/dvB)        
    tsnrB = np.copy(snrB)*np.sqrt(2*smoothnB)
    tfluxB = np.copy(sfluxB) - t/np.copy(tsnrB)
    sfluxB = smoothSpectra(sfluxB,smoothnB)
    sfluxB[tfluxB<0] = 0 
            
    sigmaT = np.mean(t/np.copy(tsnrB))
    
    plots = 0
    if plots:
        plt.plot(vsB,fluxB)
        plt.plot(vsB,sfluxB)
        plt.plot(vsB,sfluxB*0)
        plt.plot(vsB,3/tsnr)
        plt.show(block=False)
        input('Enter something to continue...')
        plt.close()

    if testingStack:
        fluxB = createTestFlux(fluxB)
        sfluxB = np.copy(fluxB)
        print np.shape(fluxB)

    #ok, so now we actually need to iterate through spectra, stack
    stackCount = 0
    bigStackCount = 0
    darkGap = 0
    darkIndexReset = -999
    darkIndex = darkIndexReset
    nullValue = -999
    vrange = 1000 #km/s
    nrange = np.ceil(vrange/dvA) #number of pixels to stack
    #we'll eventually need to interpolate onto a fixed grid here
    vstack = np.arange(0,5000,dvA)
    nrange = np.size(vstack)
    nrangeB = np.round(nrange*dvA/dvB)
    vgrid = np.arange(0,1000,2)
    smallCutoff = smallCutoff/dvB
    largeCutoff = largeCutoff/dvB
    superSmallCutoff = superSmallCutoff/dvB
    
    smallStackP = nullValue*np.ones(np.shape(vgrid))
    smallStackM = nullValue*np.ones(np.shape(vgrid))
    largeStackP = nullValue*np.ones(np.shape(vgrid))
    largeStackM = nullValue*np.ones(np.shape(vgrid))
    fluxlength = np.size(fluxB)
    if stackPlot:
        #plt.plot(vsB,sfluxB)
        plt.plot(vsB,fluxB)
        plt.xlabel('v (km/s)')
    #plt.ylabel('F_{unsmoothed}(v)')

    zminA = np.min(zsA)
    zmaxA = np.max(zsA)
    
    for i in range(0,fluxlength):
        if (sfluxB[i] == 0):
            darkGap = darkGap + 1
            #mark beginning of dark gap
            if (darkIndex == darkIndexReset):
                darkIndex = i
        if ((sfluxB[i] != 0) & (sfluxB[i-1]==0) & (i > 0)):
            #print "dark gap is %d pixels..." % (darkGap)
            # OK, so maybe we should first check if the dark 
            # gap has Lya coverage.
            # ---> If so, is it absorbed in Lya? Make this
            #      a requirement. Call this doubleDark
            doubleDark = 1
            zDGmin = zsB[darkIndex]
            zDGmax = zsB[i]
            if (zDGmin > zDGmax):
                temp = np.copy(zDGmin)
                zDGmin = np.copy(zDGmax)
                zDGmax = np.copy(temp)
            # check for any overlap with Lya
            condition1 = zmaxA > zDGmin
            condition2 = zminA < zDGmax
            if (condition1&condition2):
                # we have some overlap
                # find largest z of overlap
                zupper = np.min((zDGmax,zmaxA))
                zlower = np.max((zDGmin,zminA))
                miniA = np.size(zsA[zsA<=zlower])-1
                maxiA = np.size(zsA[zsA<=zupper])-1
                
                if (np.max(sfluxA[miniA:maxiA]) > sigmaT/2):
                    doubleDark = 0
                    print "failed to find simultaneous Lya"
                    print "and Lyb absorption"          
                    print np.max(sfluxA[miniA:maxiA]),sigmaT
                    print miniA,darkIndex,maxiA,i
            
            if (requireLyab == 0):
                doubleDark = 1
            # proceed with stacking if no overlap 
            # or if overlap is absorbed
            if doubleDark:
                # stack in one direction, check for boundaries too
                if (darkGap >= largeCutoff):
                    # print "dark gap is %d pixels..." % (darkGap)
                    stackCount = stackCount + 1
                    # OK, so here we probably have to do some thinking
                    mini = i
                    maxi = i + nrangeB
                    plus = nullValue*np.ones(nrange)
                    if (maxi >= fluxlength):
                        maxi = (fluxlength - 1)
                    zminB = zsB[mini]
                    zmaxB = zsB[maxi]


                    # check if able to stack
                    if ((zminB>=zminA)&(zminB<zmaxA)):
                        miniA = np.size(zsA[zsA<=zminB])-1
                        maxiA = miniA + nrange 
                        beginning = 0
                        if (maxiA >= np.size(fluxA)):
                            maxiA = np.size(fluxA) - 1
                        nextent = maxiA-miniA
                 
                    else:
                        nextent = 0
                        print "should this error have happened?"
                

                    # if (nextent>=nrange):
                    #    nextent = nrange-1
                    if (nextent > 0):
                        condition1 = (zcut>0)&(zsB[mini]>=zcut)
                        condition2 = (zcut<0)&(zsB[mini]<=np.abs(zcut))
                    
                        if (condition1 | condition2):
                            plus[0:nextent] = fluxA[miniA:maxiA]
                            ps = sp.interp1d(vstack,plus)
                            plusgrid = ps(vgrid)
                            #print vstack, plus, vgrid, plusgrid                        
                            largeStackP = np.vstack((largeStackP,plusgrid))
                            Ldark = dvB*np.abs(darkIndex-i)
                            Lt = t/snrB[i]
                            addPoint = np.array([Lt,Ldark])
                            Larray = np.vstack((Larray,addPoint))
                            bigStackCount = bigStackCount + 1
                            if stackPlot:
                                plt.plot(np.array([vsB[mini],vsB[mini]]),np.array([0,1]))
                                plt.plot(np.array([vsB[maxi],vsB[maxi]]),np.array([0,1]))
                                #print "I made a stack I made a stack!"
                                plt.show(block=False)
                            # negative stack
                    mini = darkIndex - nrangeB - 1
                    maxi = darkIndex - 1
                    minus = nullValue*np.ones(nrange)
                    if (mini < 0):
                        mini = 0
                
                    zminB = zsB[mini]
                    zmaxB = zsB[maxi]

                    # check if able to stack
                    if ((zmaxB>zminA)&(zmaxB<=zmaxA)):
                        maxiA = np.size(zsA[zsA<=zmaxB])-1
                        miniA = maxiA - nrange 
                        beginning = 1
                        if (miniA < 0):
                            miniA = 0
                        nextent = maxiA-miniA
                    else:
                        nextent = 0
                        print "should this error have happened?"
            
                    if (darkIndex <= 0):
                        nextent = 0


                    # if (nextent >= nrange):
                    #    nextent = nrange-1
                    # something somewhat-complicated will happen here
                    if (nextent > 0):
                        condition1 = (zcut>0)&(zsB[maxi]>=zcut)
                        condition2 = (zcut<0)&(zsB[maxi]<=np.abs(zcut))
                        if (condition1 | condition2):
                            minus[-1*nextent::] = fluxA[miniA:maxiA]
                            minus = minus[::-1]
                            ms = sp.interp1d(vstack,minus)
                            minusgrid = ms(vgrid)
                            # print vstack, minus, vgrid, minusgrid
                            largeStackM = np.vstack((largeStackM,minusgrid))
                            Ldark = dvB*nextent
                            Lt = t/snrB[i]
                            if stackPlot:
                                plt.plot(np.array([vsB[maxi],vsB[maxi]]),np.array([0,1]))
                                plt.plot(np.array([vsB[mini],vsB[mini]]),np.array([0,1]))
                                plt.show(block=False)
                                # print "I made a stack I made a stack!"
                
                if ((darkGap < smallCutoff) & (darkGap >= superSmallCutoff)):
                    # print "dark gap is %d pixels..." % (darkGap)
                    stackCount = stackCount + 1
                    # contributes to small stack
                    # need to stack at both edges

                    # positive stack
                    mini = i
                    maxi = i + nrangeB
                    plus = nullValue*np.ones(nrange)
                    if (maxi >= fluxlength):
                        maxi = (fluxlength - 1)
                    zminB = zsB[mini]
                    zmaxB = zsB[maxi]
                
                    # check if able to stack
                    if ((zminB>=zminA)&(zminB<zmaxA)):
                        miniA = np.size(zsA[zsA<=zminB])-1
                        maxiA = miniA + nrange 
                        beginning = 1
                        if (maxiA >= np.size(fluxA)):
                            maxiA = np.size(fluxA) - 1
                        nextent = maxiA-miniA
                    else: 
                        nextent = 0
                        print "should this error have happened?"                        
                    # if (nextent >= nrange):
                    #    nextent = nrange - 1
                    if (nextent > 0):
                        condition1 = (zcut>0)&(zsB[mini]>=zcut)
                        condition2 = (zcut<0)&(zsB[mini]<=np.abs(zcut))
                        if (condition1 | condition2):
                            plus[0:nextent] = fluxA[miniA:maxiA]
                            ps = sp.interp1d(vstack,plus)
                            plusgrid = ps(vgrid)
                            smallStackP = np.vstack((smallStackP,plusgrid))
                            Ldark = dvB*np.abs(darkIndex-i)
                            Lt = t/snrB[i]
                            addPoint = np.array([Lt,Ldark])
                            Larray = np.vstack((Larray,addPoint))
                            if stackPlot:
                                plt.plot(np.array([vsB[mini],vsB[mini]]),np.array([0,1]),'--')
                                plt.plot(np.array([vsB[maxi],vsB[maxi]]),np.array([0,1]),'--')
                                plt.show(block=False)
                                # print "I made a stack I made a stack!"

                    # negative stack
                    mini = darkIndex - nrangeB - 1
                    maxi = darkIndex - 1
                    minus = nullValue*np.ones(nrange)
                    if (mini < 0):
                        mini = 0
                    
                    zminB = zsB[mini]
                    zmaxB = zsB[maxi]
                
                    if ((zmaxB>zminA)&(zmaxB<=zmaxA)):
                        maxiA = np.size(zsA[zsA<=zmaxB])-1
                        miniA = maxiA - nrange 
                        beginning = 1
                        if (miniA < 0):
                            miniA = 0
                        nextent = maxiA-miniA
                    else:
                        nextent = 0
                        print "should this error have happened?"

                    if (darkIndex <= 0):
                        nextent = 0

                    if (nextent > 0):
                        condition1 = (zcut>0)&(zsB[maxi]>=zcut)
                        condition2 = (zcut<0)&(zsB[maxi]<=np.abs(zcut))
                        if (condition1 | condition2):
                            minus[-1*nextent::] = fluxA[miniA:maxiA]
                            minus = minus[::-1]
                            ms = sp.interp1d(vstack,minus)
                            minusgrid = ms(vgrid)
                            smallStackM = np.vstack((smallStackM,minusgrid))
                            Ldark = dvB*nextent
                            Lt = t/snrB[i]
                            if stackPlot:
                                plt.plot(np.array([vsB[maxi],vsB[maxi]]),np.array([0,1]),'--')
                                plt.plot(np.array([vsB[mini],vsB[mini]]),np.array([0,1]),'--')
                                plt.show(block=False)                 
                                # print "I made a stack I made a stack!"
            darkGap = 0
            darkIndex = darkIndexReset

    plt.show(block=False)
    #input('enter something to continue to next spectrum...')
    fluxvar = np.var(fluxA)
    print "%s contributed %d stacks..." % (mFilenameA,stackCount)
    print "%s contributed %d large dark gaps..." % (mFilenameA,bigStackCount)
    return largeStackP,largeStackM,smallStackP,smallStackM,meansnrA,Larray,bigStackCount,sigmaT




def averageStacks(LP,LM,SP,SM,WLP,WLM,WSP,WSM):
    #LM = np.fliplr(LM)
    #SM = np.fliplr(SM)
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

    # plt.plot(WL**2)
    # plt.plot(WS**2)
    #print np.min(WL**2),np.max(WL**2)
    #plt.title('Inverse-Variance Weighting (Estimated SNRs)')
    #plt.ylabel('Weighting (Arbitrary Units)')
    #plt.xlabel('Contribution Number')
    #plt.legend(['L > 500 km/s','L < 300 km/s'])
    #plt.show(block=False)    
    #input("press return to continue...")
    #plt.close()
    
    # print "disabling inverse-variance weighting..."
    # WL = np.ones(np.shape(WL))
    # WS = np.ones(np.shape(WS))
    
    print "shape of W matrix:"
    print np.shape(WL),np.shape(WS)
    print np.shape(L),np.shape(S)
    maxflux = 2
    epsilon = 1e-10
    #count how many stacks contribute to each bin
    Lcounts = np.zeros(np.shape(L)[1])+epsilon
    Scounts = np.zeros(np.shape(L)[1])+epsilon
    weightHist = np.array(())
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
                # weightHist = np.append(weightHist,WL[i])
    large = large/Lcounts

    ## Calculate Small Dark Gap Means
    for i in range(0,np.shape(S)[0]):
        for j in range(0,np.shape(S)[1]):
            if (np.abs(S[i,j]) < maxflux):
                Scounts[j] = Scounts[j] + WS[i]
                small[j] = small[j] + S[i,j]*WS[i]
                # weightHist = np.append(weightHist,WS[i])
    small = small/Scounts
    
    ## Calculate Large Dark Gap Variances
    for i in range(0,np.shape(L)[0]):
        for j in range(0,np.shape(L)[1]):
            if (np.abs(L[i,j]) < maxflux):
                # largeVar[j] = largeVar[j] + (L[i,j]-large[j])**2
                largeVar[j] = largeVar[j] + WL[i]*(L[i,j]-large[j])**2
    largeVar = largeVar/Lcounts**2
    largeVar = 1/np.sqrt(Lcounts)
    ## Calculate Small Dark Gap Variances
    for i in range(0,np.shape(S)[0]):
        for j in range(0,np.shape(S)[1]):
            if (np.abs(S[i,j]) < maxflux):
                # smallVar[j] = smallVar[j] + (S[i,j]-small[j])**2
                smallVar[j] = smallVar[j] + WS[i]*(S[i,j]-small[j])**2
    smallVar = smallVar/Scounts**2
    smallVar = 1/np.sqrt(Scounts)


    plots = 0
    if plots:
        weightHist = np.hstack((WS,WL))
        print np.shape(weightHist)
        plt.hist(weightHist,bins=30)
        plt.title('Distribution of Stack Weightings')
        plt.xlabel('Weighting')
        plt.ylabel('Frequency')
        plt.show(block=False)
        input("Press Return to Continue...")
        plt.figure
        print "YOUR VARIANCE CALCULATION IS DUBIOUS!!!!"
        print "YOUR VARIANCE CALCULATION IS DUBIOUS!!!!"
        print "YOUR VARIANCE CALCULATION IS DUBIOUS!!!!"
        print "YOUR VARIANCE CALCULATION IS DUBIOUS!!!!"
        print "YOUR VARIANCE CALCULATION IS DUBIOUS!!!!"
    return large,small,largeVar,smallVar



def smoothSpectra(fux,n):
    #we want to just smooth this in a natural way and we are 
    #frustrated after looking to Google for help
    fluxcopy = np.copy(fux)
    maxIndex = len(fluxcopy)-n
    for i in range(n.astype(int),maxIndex.astype(int)):
        fluxcopy[i] = np.mean(fluxcopy[i-n:(i+n+1)])
    
    #take care of edges
    fluxcopy[0:n] = np.mean(fluxcopy[0:n])
    fluxcopy[-n::] = np.mean(fluxcopy[-n::])

    return fluxcopy


def stackMatrices(lmin,lmax,zcut,t,UseLyb):
    #f,z = getFilenames()
    # This will include files without real noise estimates
    f,fbase,z = getAllFilenames()
    fb,fbaseb,zb = getAllFilenamesLyb()
    PLfit = 0
    if (UseLyb == 0):
        fb = f
    nSpectra = len(f)
    Larray = np.array([1,1])
    for i in range(0,nSpectra):
        if (PLfit):
            fileBase, fileExt = os.path.splitext(f[i])
            finput = "%s_PLfit%s" % (fileBase,fileExt)
            fileBase, fileExt = os.path.splitext(fb[i])
            fbinput = "%s_PLfit%s" % (fileBase,fileExt)
        else:
            finput = f[i]
            fbinput = fb[i]
        if (i != 4):
            LP,LM,SP,SM,snr,Larray,dv = oneStackBinZLyb(finput,fbinput,z[i],lmin,lmax,zcut,t,Larray)
            # LP,LM,SP,SM,snr,Larray = oneStackBinZ(f[i],z[i],lmin,lmax,zcut,t,Larray)
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

                LPindex = np.zeros(np.shape(LP)[0])
                LMindex = np.zeros(np.shape(LM)[0])
                SPindex = np.zeros(np.shape(SP)[0])
                SMindex = np.zeros(np.shape(SM)[0])

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

                    if (np.size(np.shape(LP)) == 2):
                        LPindexT = i*np.ones(np.shape(LP)[0])
                    else:
                        LPindexT = i    
                    LPindex = np.hstack((LPindex,LPindexT))


                    if (np.size(np.shape(LM)) == 2):
                        LMindexT = i*np.ones(np.shape(LM)[0])

                    else:
                        LMindexT = i
                    LMindex = np.hstack((LMindex,LMindexT))                    
                    if (np.size(np.shape(SP)) == 2):
                        SPindexT = i*np.ones(np.shape(SP)[0])
                    else:
                        SPindexT = i
                    SPindex = np.hstack((SPindex,SPindexT))
                        
                    if (np.size(np.shape(SM)) == 2):
                        SMindexT = i*np.ones(np.shape(SM)[0])
                    else: 
                        SMindexT = i
                    SMindex = np.hstack((SMindex,SMindexT))
                    
                    
                
                    
                    
                    
                    print np.shape(LPindex),np.shape(LPfull),np.size(np.shape(LP)),np.size(LP)


    large,small,largeVar,smallVar = averageStacks(LPfull,LMfull,SPfull,SMfull,WLPfull,WLMfull,WSPfull,WSMfull)        
    vs = 2*np.arange(0,np.shape(large)[0],1)
    #print np.shape(LPfull),np.shape(SPfull)
    return vs,LPfull,LMfull,SPfull,SMfull,large,small,LPindex,LMindex

def zsToVs(z,zs):
    lambdaA0 = 1216*(1+z)
    lambdas = 1216*(1+zs)
    c = 3e5
    vs = c*(lambdaA0-lambdas)/lambdaA0
    return vs

def createTestFlux():
    # this program will generate flux files with names
    # similar to the actual spectra, but where we know
    # what their stacked behavior should be.

    f,fbase,z = getAllFilenames()
    fb,fbaseb,zb = getAllFilenamesLyb()
    nspectra = len(f)
    for i in range(0,nspectra):
        data = np.genfromtxt(f[i])
        zs = data[0,:]
        snrs = data[2,:]
        transmission = np.ones(np.shape(zs))
        testflux = transmission - np.exp(-(zs-np.mean(zs))**2/.005)
        testflux[testflux<.2] = 0
        fileBase, fileExt = os.path.splitext(f[i])
        foutput= "%s_testflux.tex" % (fileBase)
        fileBase, fileExt = os.path.splitext(fb[i])
        fboutput = "%s_testflux.tex" % (fileBase)

        LyaMatrix = np.vstack((zs,testflux,snrs))
        LybMatrix = np.vstack((zs,testflux,snrs))
        np.savetxt(foutput, LyaMatrix, delimiter = "\t")
        np.savetxt(fboutput, LybMatrix, delimiter = "\t")
        print "finished spectrum %d..." % (i)


def fullStack(lmin,lmax,zcut,t,UseLyb):
    #f,z = getFilenames()
    # This will include files without real noise estimates
    f,fbase,z = getAllFilenames()
    fb,fbaseb,zb = getAllFilenamesLyb()
    PLfit = 0
    commonRes = 1
    testingStack = 0

    if (UseLyb == 0):
        fb = f
    nSpectra = len(f)
    Larray = np.array([1,1])
    sindex = np.arange(0,nSpectra,1)
    scounts = np.zeros(np.shape(sindex))
    tcounts = np.zeros(np.shape(sindex))
    zcounts = np.zeros(np.shape(sindex))
    wcounts = np.zeros(np.shape(sindex))
    for i in range(0,nSpectra):
        if commonRes:
            fileBase, fileExt = os.path.splitext(f[i])
            finput= "%s_smoothed.tex" % (fileBase)
            fileBase, fileExt = os.path.splitext(fb[i])
            fbinput = "%s_smoothed.tex" % (fileBase)
        else:
            finput = f[i]
            fbinput = fb[i]
        if (PLfit):
            fileBase, fileExt = os.path.splitext(f[i])
            finput = "%s_PLfit%s" % (fileBase,fileExt)
            fileBase, fileExt = os.path.splitext(fb[i])
            fbinput = "%s_PLfit%s" % (fileBase,fileExt)
        if testingStack:
            fileBase, fileExt = os.path.splitext(f[i])
            finput= "%s_testflux.tex" % (fileBase)
            fileBase, fileExt = os.path.splitext(fb[i])
            fbinput = "%s_testflux.tex" % (fileBase)

        if (i != 2):
            fileBase, fileExt = os.path.splitext(f[i])
            weightFile = "%s_weighting%s" % (fileBase,fileExt)
            LP,LM,SP,SM,snr,Larray,stackCount,sigmaT = oneStackBinZLyb(finput,fbinput,z[i],lmin,lmax,zcut,t,Larray)
            scounts[i] = stackCount
            tcounts[i] = sigmaT
            zcounts[i] = z[i]
            if (i == 10):
                print f[i]
                print "THAT ONE HAD THE MOST!!!!"
            
            # LP,LM,SP,SM,snr,Larray = oneStackBinZ(f[i],z[i],lmin,lmax,zcut,t,Larray)
            ### fill in weighting matrix
            print "SNR = %f"%(snr)
            #snr = 1
            fileBase, fileExt = os.path.splitext(fbase[i])
            snrFile ="%s_snrEstimate%s" % (fileBase, fileExt)
            
            #snr = np.genfromtxt(snrFile)
            weighting = np.genfromtxt(weightFile)
            # print "snr = %f" % (snr)            
            # snr = 1/np.sqrt(fluxvar)
            #snr = snr/np.sqrt(dv/2)
            print "weighting = %f" % (weighting)
            wcounts[i] = weighting
            #print "smoothing factor: %f" % (np.sqrt(dv/2))
            #print "dv = %f km/s" % (dv)
            #print "disabling weighted averaging..."
            if (np.size(np.shape(LP)) == 2):
                WLP = weighting*np.ones(np.shape(LP)[0])
            else:
                WLP = weighting
            if (np.size(np.shape(LM)) == 2):
                WLM = weighting*np.ones(np.shape(LM)[0])
            else:
                WLM = weighting
            if (np.size(np.shape(SP)) == 2):
                WSP = weighting*np.ones(np.shape(SP)[0])
            else:
                WSP = weighting
            if (np.size(np.shape(SM)) == 2):
                WSM = weighting*np.ones(np.shape(SM)[0])
            else: 
                WSM = weighting

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
        plt.errorbar(vs,large,yerr=largeErr)
        plt.xlabel('v (km/s)')
        plt.show(block=False)
        input("Enter something to continue (This is large stack):")
        plt.plot(vs,small)
        smallErr = np.sqrt(smallVar)
        plt.errorbar(vs,small,yerr=smallErr)
        plt.legend(('Large Stack','Small Stack'))
        plt.show(block=False)
        input("Now we have added the small stack")
        plt.close()
        plt.plot(vs,largeErr)
        plt.plot(vs,smallErr)
        plt.show(block=False)
        input("these are the errors?")

    # plt.plot(zcounts,tcounts,'x')
    # plt.plot(zcounts,scounts,'o')
    # plt.plot(zcounts,wcounts,'v')
    # plt.legend(['Threshold','Contributions','Weightings'])
    # plt.show(block=False)
    
    initialStackGuess = np.sum(tcounts*scounts*wcounts)/np.sum(wcounts*scounts)
    print "initial stack flux guess is %f" % (initialStackGuess)
    #input('do something to continue...')
    plots = 0
    if plots:
        plt.plot(zcounts[zcounts!=0],scounts[zcounts!=0],'o')
        # plt.plot(sindex,scounts,'o')
        plt.xlabel('Spectrum Redshift')
        # plt.ylabel('t$\sigma_{N}$')
        plt.ylabel('Number of Contributions to Stack (L>1200km/s)')
        plt.ylabel('Weighting of Spectrum')
        plt.show(block=False)
        input("Enter something to continue...")



    Lscatter = 0
    if Lscatter:
        print np.shape(Larray)
        Lt = Larray[:,0]
        Ldark = Larray[:,1]
        plt.semilogy(Lt,Ldark,'x')
        plt.xlabel('Smoothed Flux Threshold')
        plt.ylabel('Dark Gap Size (km/s)')
        plt.show(block=False)
        input("Press return to end program...")
    return vs,large,small,largeVar,smallVar


def FluxVarEstimate():
    lmin = 500 
    lmax = 300
    zcut = 5
    t = 3
    UseLyb = 0

    f,fbase,z = getAllFilenames()
    fb,fbaseb,zb = getAllFilenamesLyb()
    PLfit = 0
    if (UseLyb == 0):
        fb = f
    nSpectra = len(f)
    
    for i in range(0,nSpectra):
        lambdaA0 = 1216*(1+z[i])
        c = 3e5
        mdata = np.genfromtxt(f[i])
        zs = mdata[0,:]
        lambdasA = 1216*(1+zs)
        flux = np.copy(mdata[1,:])
        snr = np.copy(mdata[2,:])
    
        vs = c*(lambdaA0-lambdasA)/lambdaA0
        dv = np.abs(vs[1]-vs[0])

        fluxvar = np.var(flux)
    
        fileBase, fileExt = os.path.splitext(fbase[i])
        snrFile ="%s_snrEstimate%s" % (fileBase, fileExt)          
        snrRed = np.genfromtxt(snrFile)
        print "snrRed = %f, snr = %f" % (snrRed,np.mean(snr))            
        # snr = 1/np.sqrt(fluxvar)
        # snr = snr/np.sqrt(dv/2)
        
        sigmaF = fluxvar-1/np.mean(snr)**2
        if (sigmaF < 0):
            print "filename: %s" % (f[i])
            print "dv = %f, sigmaF = %f, sigmaT = %f, sigmaN = %f" % (dv,sigmaF,fluxvar,1/np.mean(snr**2))        
        # sigmaF = np.sqrt(fluxvar**2-1/snrRed**2)
        
        if (np.abs(np.mean(snr)-snrRed) > .1):
            # plt.plot(dv,fluxvar-1/np.mean(snr**2),'x')
            sigmaF = np.sqrt(fluxvar-1/np.mean(snr**2))
            plt.plot(dv,sigmaF,'x')
            print "dv: %f, sigmaF: %f" % (dv,sigmaF)
            plt.xlabel('$\Delta$ v (km/s)')
            plt.ylabel('$\sigma_{F}$')
            plt.show(block=False)
            print "filename: %s" % (f[i])
            print "plotting..."
        

    input("press return to end...")

def StackVaryL():
    
    lmin = np.array([100,300,500,1000])
    lmax = 300
    zcut = 5
    t = 3
    UseLyb = 1
    for i in range(0,np.size(lmin)):
        vs,large,small,largeVar,smallVar = fullStack(lmin[i],lmax,zcut,t,UseLyb)
        if (i == 0):
            plt.plot(vs,small,'--')
            smallErr = np.sqrt(smallVar)
            #plt.errorbar(vs,small,yerr=smallErr)
        plt.plot(vs,large)
        plt.plot(vs,large)
        largeErr = np.sqrt(largeVar)
        #plt.errorbar(vs,large,yerr=largeErr)
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

def StackOneL():
    
    lmax = 300
    lmin = 500
    UseLyb = 0
    zcut = 5
    t = 3
    vs,large,small,largeVar,smallVar = fullStack(lmin,lmax,zcut,t,UseLyb)
    

    smallErr = np.sqrt(smallVar)
    largeErr = np.sqrt(largeVar)
    # plt.errorbar(vs,small,yerr=smallErr)
    # plt.errorbar(vs,large,yerr=largeErr)
    plt.plot(vs,small,'--')
    plt.plot(vs,large)
    plt.legend('Small')
    plt.xlabel('v (km/s)')
    plt.ylabel('Stacked Transmission')
    plt.title('Splitting at z = %f, Using Lyb = %d'%(zcut,UseLyb))
    legend1 = "L < %d km/s" % (lmax)
    legend2 = "L > %d km/s" % (lmin)
    plt.legend((legend1,legend2))


    plt.show(block=False)
    input("Press return to end...")


def StackVaryThreshold():
    
    ts = np.array([3,5,900])
    lmin = 500
    lmax = 500
    zcut = 5
    UseLyb = 1
    for i in range(0,np.size(ts)):
        vs,large,small = fullStack(lmin,lmax,zcut,ts[i],UseLyb)

        plt.plot(vs,small,'--')
        plt.plot(vs,large)
        plt.legend('Small')
        plt.xlabel('v (km/s)')
        plt.ylabel('Stacked Transmission')
        plt.title('Splitting at z = %f, $L_{min}$ = %d km/s'%(zcut,lmin))
        legend1 = "threshold = %f $\sigma$" % (ts[0])
        legend2 = "threshold = %f $\sigma$" % (ts[0])
        legend3 = "threshold = %f $\sigma$" % (ts[1])
        legend4 = "threshold = %f $\sigma$" % (ts[1])
        legend5 = "threshold = %f $\sigma$" % (ts[2])
        legend6 = "threshold = %f $\sigma$" % (ts[2])

        plt.legend((legend1,legend2,legend3,legend4,legend5,legend6))
    plt.show(block=False)
    input("Press return to end...")


def gridPlot():
    
    tarray = np.array([1,3,5])
    zarray = np.array([5,5.25,5.5,5.75])
    nt = np.size(tarray)
    nz = np.size(zarray)

    UseLyb = 0
    for i in range(0,nt):
        for j in range(0,nz):
            
            
            ax1 = plt.subplot2grid((nt,nz), (i,j))
            ### We will run this for each iteration of the loop
            lmin = np.array([100,300,500,1000])
            lmax = 300
            zcut = zarray[j]
            for q in range(0,np.size(lmin)):
                vs,large,small,largeVar,smallVar = fullStack(lmin[q],lmax,zcut,tarray[i],UseLyb)
                if (q == 0):
                    plt.plot(vs,small,'--')
                plt.plot(vs,large)
                #plt.legend('Small')
                if (i==(nt-1)):
                    plt.xlabel('v (km/s)')
                if (j==0):
                    plt.ylabel('t = %d $\sigma$'%(tarray[i]))
                if (i==0):
                    plt.title('z = %f, Use Lyb = %d'%(zcut,UseLyb))
                legend1 = "L < %d km/s" % (lmax)
                legend2 = "L > %d km/s" % (lmin[0])
                legend3 = "L > %d km/s" % (lmin[1])
                legend4 = "L > %d km/s" % (lmin[2])
                legend5 = "L > %d km/s" % (lmin[3])
                #plt.legend((legend1,legend2,legend3,legend4,legend5))
            plt.show(block=False)
    input("Press return to end...")


def stepByStep():
    # This will show, step by step, how the stacking proceeds.
    lmin = 300
    lmax = 500
    UseLyb = 1
    t = 3
    zcut = 4.5
    vs,LP,LM,SP,SM,large,small,LPindex,LMindex = stackMatrices(lmin,lmax,zcut,t,UseLyb)
    LM = np.fliplr(LM)
    SM = np.fliplr(SM)
    L = np.vstack((LP,LM))
    S = np.vstack((SP,SM))
    Lindex = np.hstack((LPindex,LMindex))
    print "shape of index vector:"
    print np.shape(Lindex),np.shape(L)
    nstacks = np.shape(L)[0]
    spectraPerPlot = 5
    onevs = vs[:]
    j = 1
    for q in range(0,spectraPerPlot-1):
        print np.shape(vs)
        vs = np.vstack((vs,onevs))
        print np.shape(vs)
    for i in range(0,nstacks):
        
        
        mini = i*spectraPerPlot
        maxi = mini+spectraPerPlot
        print np.shape(vs),np.shape(L[mini:maxi,:])
        tempL = L[i,:]
        plt.plot(onevs[tempL>-10000],tempL[tempL>-10000])
        print "j,i,Lindex[i]:"
        print j,i,Lindex[i]
        # plt.plot(L[mini:maxi,:])
        
        if (np.mod(i,spectraPerPlot-1)==0):
            plt.plot(onevs,large)
            plt.axis([0,800,-.2,1])
            plt.xlabel('$\Delta$ v (km/s)')
            plt.ylabel('F')
            plt.title('Contributions to Stacked Transmission (Frame %d)' % (j))
            
            plt.legend(['1','2','3','4','5','6'])
            plt.show(block=False)
            # input('Press return to continue...')
            F = p.gcf()
            filename = "StepByStep_Frame%03d.png" % (j)
            j = j + 1
            F.savefig(filename, bbox_inches='tight')
            plt.close()
            


if __name__ == "__main__":
    
    # stepByStep()
    # StackOneL()
    # FluxVarEstimate()
    # findWeightings()
    gridPlot()
    # createTestFlux()
