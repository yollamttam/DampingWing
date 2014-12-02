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

def oneStack(mFilename,z):
    print mFilename
    mdata = np.genfromtxt(mFilename)
    zs = mdata[0,:]
    flux = mdata[1,:]
    snr = mdata[2,:]

    #create variable that is velocity separation 
    c = 3e5
    lambdaA0 = 1216*(1+z)
    lambdas = 1216*(1+zs)
    vs = c*(lambdaA0-lambdas)/lambdaA0

    #create smoothed version of flux. 
    smoothV = 50.0
    dv = np.abs(vs[1]-vs[0])
    print "apparently, resolution is %f km/s..."%(dv)
    smoothn = np.ceil(smoothV/dv)
    print "smoothing over %f pixels"%(smoothn)
    sflux = smoothSpectra(flux,smoothn)
    tsnr = snr/np.sqrt(2*smoothn)
    tflux = sflux - 3*tsnr
    print "the above thing is wrong..."
    sflux[tflux<0] = 0


    plt.plot(vs,flux)
    plt.plot(vs,sflux)
    plt.show(block=False)
    input('Enter something to continue...')
    plt.close()

def smoothSpectra(fux,n):
    #we want to just smooth this in a natural way and we are 
    #frustrated after looking to Google for help
    flux = fux
    maxIndex = len(flux)-n
    for i in range(n.astype(int),maxIndex.astype(int)):
        flux[i] = np.mean(flux[i-n:i+n])
    
    #take care of edges
    flux[0:n] = np.mean(flux[0:n])
    flux[-n::] = np.mean(flux[-n::])

    return flux

if __name__ == "__main__":
    f,z = getFilenames()
    nSpectra = len(f)
    for i in range(0,nSpectra):
        oneStack(f[i],z[i])
        
