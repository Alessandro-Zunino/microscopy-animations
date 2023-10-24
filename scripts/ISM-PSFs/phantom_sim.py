import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn
from numpy.random import poisson

import brighteyes_ism.simulation.Tubulin_sim as simTub
import brighteyes_ism.analysis.Graph_lib as gr
import brighteyes_ism.analysis.Tools_lib as tool

#%% read data

data = np.load('PSFs.npz')

exPSF = data['exPSF']
detPSF = data['detPSF']
PSF = data['PSF']
Nch = PSF.shape[-1]

#%% downsample PSFs

ds = 10

PSF_2 = tool.DownSample(PSF, ds = 10, order = 'xyc')
Nx = PSF_2.shape[0]
pxsizex = data['pxsizex'] * ds

gr.ShowDataset(PSF_2)

Nx_2 = Nx*11

#%%

tubulin = simTub.tubSettings()
tubulin.xy_pixel_size = pxsizex
tubulin.xy_dimension = Nx_2
tubulin.xz_dimension = 1     
tubulin.z_pixel = 1     
tubulin.n_filament = 12
tubulin.radius_filament = pxsizex*0.6
tubulin.intensity_filament = [0.5,0.9]  
phTub = simTub.functionPhTub(tubulin)

plt.figure()
plt.imshow(phTub[:,:,0],cmap='magma')
plt.axis('off')

TubDec = phTub[:,:,0]

#%%

sz = [ TubDec.shape[0], TubDec.shape[1], PSF.shape[2] ]

img = np.empty(sz)

flux = 1e3
obj = TubDec*flux

for n in range(Nch):
    img[:, :, n] = sgn.convolve(obj, PSF_2[:, :, n], mode = 'same')

#%% Convert to photons and add Poisson noise

img = np.uint16(img)

img_1 = poisson(img)
img_2 = poisson(img)

fig = gr.ShowDataset(img_1, figsize = (10,10), normalize = False )

#%%

np.savez('IMGs.npz', img = img, pxsizex = pxsizex)