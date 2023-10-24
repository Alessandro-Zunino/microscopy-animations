import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sgn

import brighteyes_ism.analysis.APR_lib as apr

import brighteyes_ism.simulation.PSF_sim as ism

import brighteyes_ism.simulation.Tubulin_sim as simTub

plt.close('all')

#%% SPAD array parameters

N = 5
Nx = 51
pxsizex = 30
pxdim = 50e3
pxpitch = 75e3
M = 500

#%% PSF simulation 2D

exPar = ism.simSettings()
exPar.wl = 640
exPar.mask_sampl = 21

emPar = exPar.copy()
emPar.wl = 680

z_shift = 0

PSF, detPSF, exPSF = ism.SPAD_PSF_2D(N, Nx, pxpitch, pxdim, pxsizex, M, exPar, emPar, z_shift=z_shift)

PSF /= np.max(PSF)

fig = plt.figure(figsize=(5,5))
for i in range(N*N):
    ax = fig.add_subplot(5, 5, i+1)
    ax.imshow(PSF[:,:,i])
    plt.axis('off')

fingerprint = ism.Fingerprint(PSF)
plt.figure();
plt.imshow(fingerprint)
plt.axis('off')

#%% Generate tubulin

# tubulin = simTub.tubSettings()
# tubulin.xy_pixel_size = pxsizex
# tubulin.xy_dimension = Nx*10
# tubulin.xz_dimension = 1     
# tubulin.z_pixel = 1     
# tubulin.n_filament = 5
# tubulin.radius_filament = pxsizex*1.0
# tubulin.intensity_filament = [0.5,0.9]  
# phTub = simTub.functionPhTub(tubulin)

# plt.figure()
# plt.imshow(phTub[:,:,0],cmap='magma')
# plt.axis('off')

# np.savez('tubulin.npz', phTub = phTub)


phTub = np.load('tubulin.npz')
phTub = phTub['phTub']


#%% decimate

TubDec = sgn.decimate( sgn.decimate(phTub[:,:,0], 10, axis = 0 ), 10, axis = 1)

TubDec[ TubDec < 0.05 ] = 0

plt.figure()
plt.imshow(TubDec,cmap='magma')
plt.axis('off')

# TubDec = phTub[:,:,0]

#%% Convolve tubulin with psf

img = np.empty(PSF.shape)

for n in range(N**2):
    img[:, :, n] = sgn.convolve(TubDec, PSF[:, :, n] ,mode = 'same')

img *= 1e3
img = np.uint16(img)

plt.figure()
plt.imshow(img[:,:,12], cmap='magma')
plt.axis('off')

img_sum = np.sum(img, axis = -1)

plt.figure()
plt.imshow(img_sum, cmap='magma')
plt.axis('off')

#%% APR

usf = 100
ref = N**2//2

shift, img_ism = apr.APR(img, usf, ref)
img_ism[img_ism<0] = 0

img_ism_sum = np.sum(img_ism, axis = -1)

plt.figure()
plt.imshow(img_ism_sum, cmap='magma')
plt.axis('off')

#%% ism scan 

filenames = []
figures = []

temp = np.zeros(PSF.shape)

path = '.\Gifs'

try:
    os.mkdir(path)
except:
    pass

lowb = np.min(img)
highb = np.max(img)

k=0

for nx in range(Nx):
    for ny in range(Nx):
        
        temp[nx, ny, :] = img[nx, ny, :]
        
        if k%5 == 0:
            
            fig, ax = plt.subplots(1,3, figsize = (10,3))
            
            ax[0].imshow(phTub[:,:,0],cmap='magma')
            ax[0].axis('off')
            ax[0].set_title('Sample')
            
            ax[0].plot(10*ny, 10*nx, color = 'c', marker ='o', markersize = 15)
            
            ax[2].imshow(temp[:,:,12], vmin = lowb, vmax = highb, cmap='magma')
            ax[2].axis('off')
            ax[2].set_title('Scanned image')
            
            ax[1].imshow(temp[nx,ny,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma')
            ax[1].axis('off')
            ax[1].set_title('Micro-image')
            
            figures.append(fig)
            
            # create file name and append it to a list
            filename = f'{path}\{nx}_{ny}.tif'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            
        k += 1

# Build gif

FPS = 15

with imageio.get_writer(path + '\ism_scan.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)

#%% test

fig = plt.figure(constrained_layout=True,figsize=(11,4))
subplots = fig.subfigures(1,3)

ax0 = subplots[0].subplots(1,1)
ax1 = subplots[1].subplots(5,5, gridspec_kw = {'wspace':0, 'hspace':0.01})
ax2 = subplots[2].subplots(1,1)

ax0.imshow(phTub[:,:,0],cmap='magma')
ax0.axis('off')
ax0.set_title('Object')

for i in range(N):
    for j in range(N):
        ax1[i,j].imshow(img[:,:,N*i+j],cmap='magma')
        ax1[i,j].axis('off')

ax1[0,2].set_title('Images')

ax2.imshow(temp[nx,ny,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma')
ax2.axis('off')
ax2.set_title('Detector')

# fig.subplots_adjust(wspace=0, hspace=0)

plt.tight_layout()

#%% ism scan 2

filenames = []
figures = []

temp = np.zeros(PSF.shape)

path = '.\Gifs'

try:
    os.mkdir(path)
except:
    pass

low = np.min(img, axis = (0,1))
high = np.max(img, axis = (0,1))

lowb = np.min(img)
highb = np.max(img)

k=0

for nx in range(Nx):
    for ny in range(Nx):
        
        temp[nx, ny, :] = img[nx, ny, :]
        
        if k%5 == 0:
            
            fig = plt.figure(constrained_layout=True,figsize=(11,4))
            subplots = fig.subfigures(1,3)
            
            ax0 = subplots[0].subplots(1,1)
            ax1 = subplots[2].subplots(5,5, gridspec_kw = {'wspace':0, 'hspace':0.01})
            ax2 = subplots[1].subplots(1,1)
            
            ax0.imshow(phTub[:,:,0],cmap='magma')
            ax0.axis('off')
            ax0.set_title('Sample')
            
            ax0.plot(10*ny, 10*nx, color = 'c', marker ='o', markersize = 15)
            
            for i in range(N):
                for j in range(N):
                    ax1[i,j].imshow(temp[:,:,N*i+j], vmin = low[N*i+j], vmax = high[N*i+j], cmap='magma')
                    ax1[i,j].axis('off')
            
            ax1[0,2].set_title('Scanned images')
            
            ax2.imshow(temp[nx,ny,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma')
            ax2.axis('off')
            ax2.set_title('Micro-image')
            
            plt.tight_layout()
            
            figures.append(fig)
            
            # create file name and append it to a list
            filename = f'{path}\{nx}_{ny}.tif'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            
        k += 1

# Build gif

FPS = 15

with imageio.get_writer(path + '\ism_scan_2.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)

#%% ism scan 3

filenames = []
figures = []

temp = np.zeros(PSF.shape)

path = '.\Gifs'

try:
    os.mkdir(path)
except:
    pass

low = np.min(img, axis = (0,1))
high = np.max(img, axis = (0,1))

lowb = np.min(img)
highb = np.max(img)

k=0

c = 0

for nx in range(Nx):
    for ny in range(Nx):
        
        temp[nx, ny, :] = img[nx, ny, :]
        
        if k%5 == 0:
            
            alpha = np.zeros( N**2 ) + 0.3
            alpha[c] = 1
            
            fig = plt.figure(constrained_layout=True,figsize=(11,4))
            subplots = fig.subfigures(1,3)
            
            ax0 = subplots[0].subplots(1,1)
            ax1 = subplots[2].subplots(5,5, gridspec_kw = {'wspace':0, 'hspace':0.01})
            ax2 = subplots[1].subplots(1,1)
            
            ax0.imshow(phTub[:,:,0],cmap='magma')
            ax0.axis('off')
            ax0.set_title('Sample')
            
            ax0.plot(10*ny, 10*nx, color = 'c', marker ='o', markersize = 15)
            
            
            for i in range(N):
                for j in range(N):
                    ax1[i,j].imshow(temp[:,:,N*i+j], vmin = low[N*i+j], vmax = high[N*i+j], cmap='magma', alpha = alpha[N*i+j])
                    ax1[i,j].axis('off')
                
            ax1[0,2].set_title('Scanned images')
            
            ax2.imshow(temp[nx,ny,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma', alpha = alpha.reshape(5,5))
            ax2.axis('off')
            ax2.set_title('Micro-image')
            
            plt.tight_layout()
            
            figures.append(fig)
            
            # create file name and append it to a list
            filename = f'{path}\{nx}_{ny}.tif'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            
            
        if k%100 == 0:
            c += 1
            c = c%(N**2)
        k += 1

# Build gif

FPS = 15

with imageio.get_writer(path + '\ism_scan_3.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)
    
#%% confocal scan 

filenames = []
figures = []

temp = np.zeros(PSF.shape)

path = '.\Gifs'

try:
    os.mkdir(path)
except:
    pass

lowb = np.min(img)
highb = np.max(img)

k=0

for nx in range(Nx):
    for ny in range(Nx):
        
        temp[nx, ny, :] = img[nx, ny, :]
        
        if k%5 == 0:
            
            fig, ax = plt.subplots(1,3, figsize = (10,3))
            
            ax[0].imshow(phTub[:,:,0],cmap='magma')
            ax[0].axis('off')
            ax[0].set_title('Sample')
            
            ax[0].plot(10*ny, 10*nx, color = 'c', marker ='o', markersize = 15)
            
            ax[2].imshow(temp[:,:,12], vmin = lowb, vmax = highb, cmap='magma')
            ax[2].axis('off')
            ax[2].set_title('Image')
            
            ax[1].imshow(temp[nx,ny,12].reshape(1,1), vmin = lowb, vmax = highb, cmap='magma')
            ax[1].axis('off')
            ax[1].set_title('Detector')
            
            plt.tight_layout()
            
            figures.append(fig)
            
            # create file name and append it to a list
            filename = f'{path}\{nx}_{ny}.tif'
            filenames.append(filename)
            
            # save frame
            plt.savefig(filename)
            plt.close()
            
        k += 1

# Build gif

FPS = 15

with imageio.get_writer(path + '\confocal_scan.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# # Remove files
for filename in set(filenames):
    os.remove(filename)
    
#%% apr scan 

filenames = []
figures = []

path = '.\Gifs'

try:
    os.mkdir(path)
except:
    pass

lowb = np.min(img)
highb = np.max(img)

k = 0

start = 3
stop = -3

for nx in range(Nx - start + stop):
    for ny in range(Nx - start + stop):
        
        if k%3 == 0:
        
            fig, ax = plt.subplots(2,2, figsize = (7,7))
        
            ax[0,0].imshow(img_sum[start:stop, start:stop], cmap='magma')
            ax[0,0].axis('off')
            ax[0,0].set_title('Before reassignment')
            
            ax[0,0].plot(ny, nx, color = 'b', marker ='o', markersize = 5)
            
            
            ax[0,1].imshow(img[nx + start, ny + start,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma')
            ax[0,1].axis('off')
            ax[0,1].set_title('Micro-image')
            
            ax[1,0].imshow(img_ism_sum[start:stop, start:stop], cmap='magma')
            ax[1,0].axis('off')
            ax[1,0].set_title('After reassignment')
            
            ax[1,0].plot(ny, nx, color = 'b', marker ='o', markersize = 5)
            
            
            ax[1,1].imshow(img_ism[nx + start, ny + start,:].reshape(N,N), vmin = lowb, vmax = highb, cmap='magma')
            ax[1,1].axis('off')
            ax[1,1].set_title('Micro-image')
            
            plt.tight_layout()
            
            figures.append(fig)
            # create file name and append it to a list
            filename = f'{path}\{nx}_{ny}.tif'
            filenames.append(filename)
            # save frame
            plt.savefig(filename)
                
            plt.close()
            
        k += 1
        
# Build gif

FPS = 10

with imageio.get_writer(path + '\ism_apr_scan.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
        
# Remove files
for filename in set(filenames):
    os.remove(filename)