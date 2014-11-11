import os
import numpy as np
import matplotlib.pyplot as plt

def plotSpectra(sFilename,z):
    print sFilename
    lambdaA = 1216*(1+z)
    lambdaB = 1026*(1+z)
    lambdaG = 972*(1+z)
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
    print [ia, ib, ig]

    fluxA = flux[ib:ia]
    fluxB = flux[ig:ib]
    vsA = vs[ib:ia]
    vsB = vs[ig:ib]

    plt.plot(vs,flux)
    plt.xlabel('$\Delta v$ (km/s)')
    plt.ylabel('Flux (Arbitrary Units)')
    plt.axis([

    plt.figure
    plt.plot(vsA,fluxA,'red')
    plt.show(block=False)

if __name__ == "__main__":
    sFilename = 'Q0836-spec.tex'
    z = 5.82
    plotSpectra(sFilename,z)

