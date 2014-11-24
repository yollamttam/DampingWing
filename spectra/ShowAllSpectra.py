import os
import numpy as np
import matplotlib.pylab as plt
import ShowSpectra


f,z = ShowSpectra.getFilenamesAndRedshifts()
nSpectra = len(f)
for i in range(0,nSpectra):
    ShowSpectra.plotSpectra(f[i],z[i])




