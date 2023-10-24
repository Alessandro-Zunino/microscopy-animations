import numpy as np
import brighteyes_ism.analysis.Graph_lib as gr
import matplotlib.pyplot as plt

import os
import imageio

plt.close('all')

#%% read data

data = np.load('IMGs.npz')

img = data['img']
pxsizex = data['pxsizex'][0]

#%%

Nx = img.shape[0]
x = np.arange(-(Nx//2), Nx//2 + 1)*pxsizex

#%% Create gif

fps = 10
duration = 1000 / fps
path = 'gif_images'

if not os.path.exists(path):
    os.mkdir(path)

#%%

filenames = []
figures = []

n = img.shape[-1]
N = np.arange(n)
D = np.hstack( [N, np.flip(N)[1:]] ).astype('int')

for k, d in enumerate(D):

    fig, ax = gr.ShowImg(img[:,:,d], pxsize_x= pxsizex*1e-3, clabel = 'Counts', cmap='hot')
    ax.set_xlim(200, 400)
    ax.set_ylim(0, 200)
    
    # create file name and append it to a list
    filename = f'{path}\{k}.tif'
    filenames.append(filename)
    
    # save frame
    plt.savefig(filename, bbox_inches = 'tight', dpi = 300)
    plt.close()
    
# Build gif
with imageio.get_writer(path + '\ISMimages-shift.gif', mode='I', duration = duration, loop = 0, subrectangles = True) as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)

# Remove files
for filename in set(filenames):
    os.remove(filename)