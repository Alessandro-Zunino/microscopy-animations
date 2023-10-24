import numpy as np
import matplotlib.pyplot as plt

import brighteyes_ism.analysis.Tools_lib as tool

import brighteyes_ism.simulation.PSF_sim as ism

#%%

N = 5 # number of detector elements in each dimension
Nx = 401 # number of pixels of the simulation space
pxsizex = 5 # pixel size of the simulation space (nm)
pxdim = 50e3 # detector element size in real space (nm)
pxpitch = 75e3 # detector element pitch in real space (nm)
M = 500 # total magnification of the optical system (e.g. 100x objective follewd by 5x telescope)

z_shift = 0 #nm

#%% PSF simulation 2D

stedPar = ism.simSettings()
stedPar.wl = 775 # depletion wavelength (nm)
stedPar.mask_sampl = 200
stedPar.mask = 'VP'

#%%

donut = ism.singlePSF(stedPar, pxsizex, Nx, z_shift = z_shift)
donut /= np.max(donut)

#%%

sat = 10

depletion = np.exp( - donut * sat )

#%%

exPar = ism.simSettings()
exPar.wl = 640 # excitation wavelength (nm)
exPar.mask_sampl = 50

emPar = exPar.copy()
emPar.wl = 660 # emission wavelength (nm)

#%%

PSF, detPSF, exPSF = ism.SPAD_PSF_2D(N, Nx, pxpitch, pxdim, pxsizex, M, exPar, emPar, z_shift=z_shift)

exPSF /= np.max(exPSF)
detPSF /= np.max(detPSF)
PSF /= np.max(PSF)

#%%

effexPSF = exPSF*depletion

#%%

import matplotlib

red = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","red"])

orange = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange"])

violet = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","violet"])


fig, ax = plt.subplots(1,3)

tool.ShowImg(fig, ax[0], donut, pxsizex, clabel='', cmap = red)

tool.ShowImg(fig, ax[1], exPSF, pxsizex, clabel='', cmap = orange)

tool.ShowImg(fig, ax[2], effexPSF, pxsizex, clabel='', cmap = violet)

#%%

np.savez('PSFs.npz', donut = donut, exPSF = exPSF, detPSF = detPSF)

#%%

donut_line = donut[Nx//2]

ex_line = exPSF[Nx//2]

effex_line = effexPSF[Nx//2]

det_line = detPSF[Nx//2,:,N**2//2+1]

plt.figure()

plt.plot(donut_line)

plt.plot(ex_line)

plt.plot(effex_line)

plt.plot(det_line)

plt.plot(det_line*effex_line)