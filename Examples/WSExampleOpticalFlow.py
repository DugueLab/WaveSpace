#%%
import sys
import os
import time

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import OpticalFlow

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
# Load some simulated data
dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/complexData")

#%%
tStart = time.time()
print("OpticalFlow started")
OpticalFlow.create_uv(waveData, 
        applyGaussianBlur=False, 
        type = "angle", 
        Sigma=1, 
        alpha = 0.1, 
        nIter = 200, 
        dataBucketName="complexData",
        is_phase = False)

print('optical flow took: ', time.time()-tStart)

trialToPlot = 4
waveData.set_active_dataBucket('UV')
ani = Plotting.plot_optical_flow(waveData, 
                                UVBucketName = 'UV',
                                PlottingDataBucketName = 'complexData', 
                                dataInds = (0, trialToPlot, slice(None), slice(None), slice(None)),
                                plotangle=True,
                                normVectorLength = True)  
output_path = os.path.join(path, "Examples/ExampleData/Output/")
ani.save( output_path + 'OpticalFlowAfterFilter_Hilbert.gif')

#%%
foi = 10
cycleLength = waveData.get_sample_rate()/ foi
freqInd = 0
motifs = hf.find_wave_motifs(waveData, 
                        dataBucketName="UV", 
                        threshold = 0.8, 
                        nTimepointsEdge=cycleLength,
                        mergeThreshold = 0.8, 
                        minFrames=cycleLength, 
                        pixelThreshold = 0.6, 
                        magnitudeThreshold=.1,
                        dataInds = (freqInd, slice(None), slice(None), slice(None), slice(None)),
                        Mask = False)


conds = waveData.get_trialInfo()
uniques = np.unique(conds)
trial_dict = {}

for trial_idx, condition in enumerate(conds):
    if condition not in trial_dict:
        trial_dict[condition] = []
    trial_dict[condition].append(trial_idx)

motifMap = np.full((len(conds),len(waveData.get_time())), -1)
for ind, motif  in enumerate(motifs):
    trial_frames_list = motif['trial_frames']
    for trial_frame in trial_frames_list:
        (trial, (start_timepoint, end_timepoint)) = trial_frame
        motifMap[trial, start_timepoint:end_timepoint] = ind

cmap = mcolors.ListedColormap(['grey', "#8F43D1", "#c50069",'#d67258', '#416ae4', '#378b8c', "#0f3200" ,'#a05195', "#4e2f13", "#3900AB","#b3ff00", "#ff0015", "#0d15ff"])
bounds = [-1, 0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
norm = mcolors.BoundaryNorm(bounds, cmap.N)
im =  plt.pcolormesh(motifMap, cmap=cmap, norm=norm)
plt.colorbar()
# One column per condition
fig, axs = plt.subplots(1, len(motifs[0:6]), figsize=(12, 6), gridspec_kw={'wspace': 0.3})

for motifInd, motif  in enumerate(motifs[0:6]):
        # Quiver plot
        axs[motifInd].quiver(-np.real(motif['average']), -np.imag(motif['average']), color='black')
        axs[motifInd].set_facecolor('white')
        axs[motifInd].set_aspect('equal')
        for spine in axs[motifInd].spines.values():
            spine.set_edgecolor(cmap(motifInd+1))
            spine.set_linewidth(2)
# %%
