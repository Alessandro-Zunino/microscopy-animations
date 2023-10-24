import numpy as np
import brighteyes_ism.analysis.Graph_lib as gr
import brighteyes_ism.simulation.PSF_sim as ism

#%%

grid = ism.GridParameters()

grid.N = [1, 9**2]
grid.pxdim = 5e3
grid.pxpitch = 3.5e3
grid.pxsizex = 3
grid.Nx = 401 
grid.M = 450

z_shift = 0 #nm

#%%

# pinhole = ism.Pinholes(grid)

# gr.ShowDataset(pinhole, gridshape = grid.N)

#%%

exPar = ism.simSettings()
exPar.wl = 640 # excitation wavelength (nm)
exPar.mask_sampl = 50

emPar = exPar.copy()
emPar.wl = 660 # emission wavelength (nm)

#%%

PSF, detPSF, exPSF = ism.SPAD_PSF_2D(grid, exPar, emPar, z_shift = z_shift)

exPSF /= np.max(exPSF)
detPSF /= np.max(detPSF)
PSF /= np.max(PSF)

#%%

gr.ShowDataset(PSF)

#%%

np.savez('PSFs.npz', exPSF = exPSF, detPSF = detPSF, PSF = PSF, pxsizex = grid.pxsizex)