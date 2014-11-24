import os
import numpy as np
import matplotlib.pyplot as plt
import pylab as p
import scipy.interpolate as sp

def getFilenamesAndRedshifts():
    nspectra = 20;
    
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
    f = np.hstack((farray1,farray2))
    
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

    return f,z

def plotSpectra(sFilename,z):
    print sFilename
    lambdaA0 = 1216
    lambdaB0 = 1026
    lambdaG0 = 972
    lambdaA = lambdaA0*(1+z)
    lambdaB = lambdaB0*(1+z)
    lambdaG = lambdaG0*(1+z)
        
    c = 3e5

    data = np.genfromtxt(sFilename)

    #check if we already know how to handle this file
    fileBase, fileExt = os.path.splitext(sFilename)
    knownFormatFile ="%s_method%s" % (fileBase, fileExt)
    knownFormat = os.path.isfile(knownFormatFile)
    if knownFormat:
        Lmethod = np.genfromtxt(knownFormatFile)    
    else:
        #prompt user for format, save information
        print "Beginning of data..."
        print data[0:10,:]
        print "End of data..."
        print data[-10:,:]
        print "Method 1: wavelength = 10^data"
        print "Method 2: wavelength = data"
        print "Method 3: wavelength = (1+z)*data"
        Lmethod = input('Method Number?:')

        outfile = open(knownFormatFile,'w')
        outfile.write("%d"%(Lmethod))
        outfile.close()

    #convert wavelengths to observed angstroms
    if (Lmethod == 1):
        lambdas = pow(10,data[:,0])
    elif (Lmethod == 2):
        lambdas = data[:,0]
    elif (Lmethod == 3):
        lambdas = (1+z)*data[:,0]

    flux = data[:,1]
    
    dlambdas = lambdas - lambdaA
    vs = c*dlambdas/lambdaA
    #relative velocities of Lyman series lines
    vA = (lambdaA - lambdaA)*c/lambdaA;
    vB = (lambdaB - lambdaA)*c/lambdaA;
    vG = (lambdaG - lambdaA)*c/lambdaA;
    #associated indices, do we need all of this?
    ia = len(vs[vs<vA])
    ib = len(vs[vs<vB])
    ig = len(vs[vs<vG])
    

    fluxA = flux[ib:ia]
    fluxB = flux[ig:ib]
    vsA = vs[ib:ia]
    vsB = vs[ig:ib]
    lambdasA = lambdas[ib:ia]
    lambdasB = lambdas[ig:ib]
    #is this the best thing to do?
    zsA = lambdasA/lambdaA0 - 1
    zsB = lambdasB/lambdaB0 - 1
    allzA = lambdas/lambdaA0 - 1
    allzB = lambdas/lambdaB0 - 1
    

    plt.plot(lambdas,flux)
    plt.plot([lambdaB, lambdaB],[min(flux), max(flux)])
    plt.plot(lambdasA,fluxA)
    plt.plot(lambdasB,fluxB)
    plt.xlabel('$\lambda$ ($\AA$)')
    plt.ylabel('Flux (Arbitrary Units)')
    plt.axis([lambdaG, lambdaA, min(fluxA), max(fluxA)])
    plt.show(block=False)

    plt.figure()
    plt.plot(zsA,fluxA)
    scaleBeta = np.mean([max(fluxA),min(fluxA)])
    scaleBetaMultiply = (max(fluxA)-min(fluxA))/(2*(max(fluxB)-min(fluxB)))
    plt.plot(zsB,scaleBetaMultiply*fluxB+scaleBeta)
    plt.legend(['Ly$\\alpha$','Ly$\\beta$'])
    plt.show(block=False)


    plt.close("all")


    ###alright, so we need to continuum fit this. 
    #Load the template
    tdata = np.genfromtxt('LBQS.lis')
    tLambdas = tdata[:,0]
    tFlux = tdata[:,1]
    plt.figure()
    plt.plot(tLambdas,tFlux)
    plt.xlabel('$\lambda$ ($\AA$)')
    plt.ylabel('Flux (Arbitrary Units)')
    plt.show(block=False)

    #wait = input('Press return to continue...')
    replot = 0
    acceptable = 0
    while (acceptable==0):
        plotarray = [1275, 1375, 0]
        while replot:
            print "Minimum/Maximum wavelengths in your spectra"
            print "are %f and %f"%(min(lambdas)/(1+z),max(lambdas)/(1+z))
            plotarray = input('[Lmin Lmax continueBool]:')
            plt.axis([plotarray[0], plotarray[1], min(tFlux), max(tFlux)])
            plt.show(block=False)
            replot = plotarray[2]
            
            
        tmin = plotarray[0]
        tmax = plotarray[1]
        #scale template flux to match spectra flux outside of Ly
        tMean = np.mean(tFlux[(tLambdas>tmin)&(tLambdas<tmax)])
        smin = tmin*(1+z)
        smax = tmax*(1+z)
        sMean = np.mean(flux[(lambdas>smin)&(lambdas<smax)])
            
        tAdjust = sMean/tMean
        tFlux = tFlux*tAdjust
        tLambdas = tLambdas*(1+z)
        
        #plot both together
        plt.figure()
        plt.plot(tLambdas,tFlux)
        plt.plot(lambdas,flux)

        maxy = max([max(fluxA),max(tFlux)])
        plt.plot([lambdaA,lambdaA],[min(fluxA),maxy])
        plt.plot([lambdaB,lambdaB],[min(fluxA),maxy])
        plt.axis([lambdaG, max(lambdas), min(fluxA), maxy])
        plt.xlabel('$\lambda$ ($\AA$)')
        plt.ylabel('Flux (Arbitrary Units)')
        plt.legend(['Template Fit','Spectra','Ly$\\alpha$','Ly$\\beta$'])
        plt.title(sFilename)
        plt.show(block=False)
        acceptable = input("Is this fit acceptable?")
        if (acceptable == 0):
            replot = 1
            
        saveBool = input("Save Figure?")
        if saveBool:
            F = p.gcf()
            saveFile ="%s.eps" % (fileBase)
            F.savefig(saveFile, bbox_inches='tight')


    ### Ok, so now we need to turn this into a transmission
    ### and then save in some common format.
    
    #ok, so first we actually need to get the lambda range
    LAmin = 1050*(1+z)
    LAmax = 1190*(1+z)
    LBmin = 980*(1+z)
    LBmax = 1020*(1+z)

    f = sp.interp1d(tLambdas,tFlux)
    LAs = lambdas[(lambdas>LAmin)&(lambdas<LAmax)]
    LBs = lambdas[(lambdas>LBmin)&(lambdas<LBmax)]
    FAs = flux[(lambdas>LAmin)&(lambdas<LAmax)]
    FBs = flux[(lambdas>LBmin)&(lambdas<LBmax)]
    Atran = FAs/f(LAs)
    Btran = FBs/f(LBs)
    zAs = allzA[(lambdas>LAmin)&(lambdas<LAmax)]
    zBs = allzB[(lambdas>LBmin)&(lambdas<LBmax)]

    plt.figure()
    plt.plot(zAs,Atran)
    plt.plot(zBs,Btran)
    
    plt.xlabel('z')
    plt.ylabel('$e^{-\tau}$')
    plt.legend(['Ly$\\alpha$','Ly$\\beta$'])
    plt.title("z = %f"%(z))
    plt.show(block=False)
    if saveBool:
        F = p.gcf()
        saveFile ="%s_Transmission.eps" % (fileBase)
        F.savefig(saveFile, bbox_inches='tight')
    #input("Enter some number to continue, guy...")

    
    print "Considering file %s" %(sFilename)
    ncol = np.shape(data)[1]
    

    outputFileA ="%s_LyaMatrix%s" % (fileBase, fileExt)
    outputFileB ="%s_LybMatrix%s" % (fileBase, fileExt)
    doneAlready = os.path.isfile(outputFileA)
    
    if (doneAlready != 1):

        if (ncol == 3):
            sigmaF = data[:,2]
        else:
            knownFormatFile ="%s_Nmethod%s" % (fileBase, fileExt)
            knownFormat = os.path.isfile(knownFormatFile)
            if knownFormat:
                method = np.genfromtxt(knownFormatFile)    
            else:
                noisefile = input('What is the corresponding noise file?')
                print "Method 1: 1/sig_F^2"
                print "Method 2: sig_F"
                print "Method 3: No Noise File..."
                method = input('Which method to extract sig_F?')
                if (method != 3):
                    skipbool = input('Skip header?')
                    if skipbool:
                        newdata = np.genfromtxt(noisefile,skip_header=1)
                    else:    
                        newdata = np.genfromtxt(noisefile)
                    nncols = np.shape(newdata)[1]
                    sigmaF = newdata[:,nncols-1]
                elif (method == 3):
                    sigmaF = np.zeros(np.shape(flux))
                outfile = open(knownFormatFile,'w')
                outfile.write("%d"%(method))
                outfile.close()
            if (method == 1):
                sigmaF = 1/np.sqrt(sigmaF)
                
        #so, what we want is something like
        #sigma_N = sigmaF/continuum
        #fortunately, we have the continuum estimate at each point.
        NAs = sigmaF[(lambdas>LAmin)&(lambdas<LAmax)]
        NBs = sigmaF[(lambdas>LBmin)&(lambdas<LBmax)]
        NtranA = NAs/f(LAs)
        NtranB = NBs/f(LBs)

        print np.shape(zAs),np.shape(Atran),np.shape(NtranA)
        #these should be the output matrices.
        # | z | transmission | sigma_N |
        LyaMatrix = np.hstack((zAs,Atran,NtranA))
        LybMatrix = np.hstack((zBs,Btran,NtranB))
    
        np.savetxt(outputFileA, LyaMatrix, delimiter=",")
        np.savetxt(outputFileB, LybMatrix, delimiter=",")

    return vs,flux




if __name__ == "__main__":
    f,z = getFilenamesAndRedshifts()
    nSpectra = len(f)
    for i in range(0,nSpectra):
        plotSpectra(f[i],z[i])

        




