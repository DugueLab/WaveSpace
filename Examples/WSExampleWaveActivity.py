#%%
# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

#%%
from WaveSpace.WaveAnalysis import WaveActivity as wa
from WaveSpace.Utils import ImportHelpers
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

#%%

waveData = ImportHelpers.load_wavedata_object("ExampleData/Output/complexData")

# %% Here we look at the spatial basis functions of our data. We index into the dataBucket to use only a subset of trials (of the same conditon) and only the frequency of interest
nBases=3
dataInd= (slice(0,1),slice(10,12),slice(None),slice(None),slice(None))
wa.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

bases = waveData.get_data('Bases')

fig, axs = plt.subplots(1, nBases, figsize=(nBases*6, 6))
if nBases == 1:
    axs = [axs]  
for b in range(nBases):
    im = axs[b].imshow(
        np.angle(bases[:, :, b]),
        cmap='hsv',
        vmin=-np.pi,
        vmax=np.pi,
        origin='lower',
        aspect='auto'
    )
    axs[b].set_title(f'wave map {b+1}')
    axs[b].set_xlabel('posy')
    axs[b].set_ylabel('posx')
    fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

plt.tight_layout()
plt.show()

#%% alternatively, we cann calculate the bases on all data at once, and sort out the weights later:
nBases=5
dataInd= None
wa.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

bases = waveData.get_data('Bases')

fig, axs = plt.subplots(1, nBases, figsize=(nBases*6, 6))
if nBases == 1:
    axs = [axs]  
for b in range(nBases):
    im = axs[b].imshow(
        np.angle(bases[:, :, b]),
        cmap='hsv',
        vmin=-np.pi,
        vmax=np.pi,
        origin='lower',
        aspect='auto'
    )
    axs[b].set_title(f'wave map {b+1}')
    axs[b].set_xlabel('posy')
    axs[b].set_ylabel('posx')
    fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

plt.tight_layout()
plt.show()

# the bases have changed to express linear combinations of all the waves we put in. 


