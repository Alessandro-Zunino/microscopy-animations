import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

import os
import imageio

#%%

red = LinearSegmentedColormap.from_list("", ["white","red"])

orange =  LinearSegmentedColormap.from_list("", ["white","C1"])

violet = LinearSegmentedColormap.from_list("", ["white","violet"])

blue = LinearSegmentedColormap.from_list("", ["white","C0"])

brown = LinearSegmentedColormap.from_list("", ["white","brown"])

#%% read data

data = np.load('PSFs.npz')

exPSF = data['exPSF']
detPSF = data['detPSF']
PSF = data['PSF']
pxsizex = data['pxsizex']

#%%

Nx = exPSF.shape[0]
x = np.arange(-(Nx//2), Nx//2 + 1)*pxsizex

#%%

ex_line = exPSF[Nx//2]

det_line = detPSF[Nx//2]

psf_line = PSF[Nx//2]

#%%

plt.figure()

plt.fill_between(x, ex_line, alpha=0.4, color = 'C0')
plt.plot(x, ex_line, color = 'C0')

plt.fill_between(x, det_line[:, 13], alpha=0.4, color = 'C2')
plt.plot(x, det_line[:, 13], color = 'C2')

plt.fill_between(x, psf_line[..., 13], alpha=0.4, color = 'C1')
plt.plot(x, psf_line[:, 13], color = 'C1')


plt.setp(plt.gca().spines.values(), linewidth=2)
plt.gca().xaxis.set_tick_params(width=2)
plt.gca().yaxis.set_tick_params(width=2)

#%% Create gif

fps = 10
duration = 1000 / fps
path = 'gif_images'

if not os.path.exists(path):
    os.mkdir(path)

#%%

filenames = []
figures = []

# n = np.sqrt( PSF.shape[-1] )
# N = np.arange(n*2, n*3)
# D = np.hstack( [N, np.flip(N)[1:]] ).astype('int')

n = PSF.shape[-1]
N = np.arange(n)
D = np.hstack( [N, np.flip(N)[1:]] ).astype('int')

for k, d in enumerate(D):

    fig = plt.figure(figsize = (8,4))

    plt.fill_between(x, ex_line, alpha=0.4, color = 'C0')
    plt.plot(x, ex_line, color = 'C0', label='Excitation PSF', linewidth = 3)

    plt.fill_between(x, det_line[:, d], alpha=0.4, color = 'C2')
    plt.plot(x, det_line[:, d], color = 'C2', label='Detection PSF', linewidth = 3)

    plt.fill_between(x, psf_line[:, d], alpha=0.4, color = 'C1')
    plt.plot(x, psf_line[:, d], color = 'C1', label='ISM PSF', linewidth = 3)
    
    plt.xticks( [ x[np.argmax(det_line[:, d])], x[np.argmax(psf_line[:, d])], 0 ], fontsize=14 )
    plt.yticks( [] )
    plt.gca().set_xticklabels( [ '$\mathregular{x_d}$', '$\mathregular{\mu}$', '0' ] )
    
    plt.legend(loc='upper left')
    plt.setp(plt.gca().spines.values(), linewidth=2)
    plt.gca().xaxis.set_tick_params(width=2)
    
    # create file name and append it to a list
    filename = f'{path}\{k}.tif'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.close()
    
# Build gif
with imageio.get_writer(path + '\ISM-PSF_v2.gif', mode='I', duration = duration, loop = 0, subrectangles = True) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)