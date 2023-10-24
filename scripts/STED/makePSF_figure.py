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

donut = data['donut']
exPSF = data['exPSF']
detPSF = data['detPSF']

#%%

sat = 10

depletion = np.exp( - donut * sat )
effexPSF = exPSF*depletion

#%%

Nx = donut.shape[0]
pxsizex = 5
x = np.arange(-(Nx//2), Nx//2 + 1)*pxsizex

donut_line = donut[Nx//2]

ex_line = exPSF[Nx//2]

effex_line = effexPSF[Nx//2]

det_line = detPSF[Nx//2, :, 13]

#%%

fig, ax = plt.subplots(1,3)

ax[0].imshow(donut, cmap = brown)
ax[0].axis('off')

ax[1].imshow(exPSF, cmap = red)
ax[1].axis('off')

ax[2].imshow(effexPSF, cmap = blue)
ax[2].axis('off')

fig.tight_layout()

plt.savefig('PSFimages.pdf', dpi = 1200)

#%%

plt.figure()

plt.fill_between(x, donut_line, alpha=0.4, color = 'brown')
plt.plot(x, donut_line, color = 'brown')

plt.fill_between(x, ex_line, alpha=0.4, color = 'red')
plt.plot(x, ex_line, color = 'red')

plt.fill_between(x, effex_line, alpha=0.4, color = 'C0')
plt.plot(x, effex_line, color = 'C0')

plt.savefig('PSFplot.pdf', dpi = 1200)

#%%

plt.figure()

plt.fill_between(x, effex_line, alpha=0.4, color = 'C0')
plt.plot(x, effex_line, color = 'C0')

plt.fill_between(x, det_line, alpha=0.4, color = 'C2')
plt.plot(x, det_line, color = 'C2')

plt.fill_between(x, det_line*effex_line, alpha=0.4, color = 'C1')
plt.plot(x, det_line*effex_line, color = 'C1')

#%% Create gif

FPS = 10
path = 'gif_images'

if not os.path.exists(path):
    os.mkdir(path)

#%%

filenames = []
figures = []


sat = np.arange(0, 1000, step = 10)/100

for s in sat:
    depletion_line = np.exp( - donut_line * s)
    effex_line = ex_line*depletion_line
    psf_line = det_line*effex_line
    
    
    fig = plt.figure(figsize = (8,4))

    plt.fill_between(x, effex_line, alpha=0.4, color = 'C0')
    plt.plot(x, effex_line, color = 'C0', label='Excitation PSF')

    plt.fill_between(x, det_line, alpha=0.4, color = 'C2')
    plt.plot(x, det_line, color = 'C2', label='Detection PSF')

    plt.fill_between(x, psf_line, alpha=0.4, color = 'C1')
    plt.plot(x, psf_line, color = 'C1', label='STED-ISM PSF')
    
    plt.xticks( [ 0, x[np.argmax(det_line)], x[np.argmax(psf_line)] ] )
    plt.gca().set_xticklabels( [ '', '$x_d$', '$\mu$' ] )
    plt.gca().text(-15, -0.04, '0')
    
    plt.xlim( [-700, 700] )
    
    plt.legend(loc='upper left')
    
    # create file name and append it to a list
    filename = f'{path}\{s}.tif'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.close()
    
# Build gif
with imageio.get_writer(path + '\STED-ISM-PSF.gif', mode='I', fps = FPS) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)