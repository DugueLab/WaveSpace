#%% Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)
#%%
from WaveSpace.Utils import ImportHelpers
from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.WaveAnalysis import WaveAnalysis as wa

import numpy as np
import matplotlib.pyplot as plt

#%%
saveFolder = "ExampleData/Output/"
waveData = ImportHelpers.load_wavedata_object(saveFolder + "SimulatedData")

#Create 10 sample points along the diagonal of the channel array
gridSize = waveData.get_data("SimulatedData").shape[1]
nPoints = range(0,gridSize,2)
sourcePointsDiagonal = []
for i in nPoints:
    sourcePointsDiagonal.append([i,i])

# # Create 10 sample points along a horizontal of the channel array
# yLocation = int(np.floor(gridSize/2))
# sourcePointsHorizontal = []
# for i in nPoints:
#     sourcePointsHorizontal.append([i,yLocation])

#restrict to (temp) frequencies between lower and upper bound:
lowerBound = 2
upperBound = 40

wa.FFT_2D(waveData, sourcePointsDiagonal, lowerBound, upperBound, DataBucketName="SimulatedData")

result = waveData.get_data("Result")

# Get the number of trials
n_trials = waveData.get_data("SimulatedData").shape[0]
trialInfo = waveData.get_trialInfo()
conditions = np.unique(trialInfo)

for condition in conditions:
    indices = [i for i, cond in enumerate(trialInfo) if cond == condition]
    logRatios=np.mean(np.log(result["Max Along Power"][indices]/result["Max Reverse Power"][indices]))
    newlineseries = np.zeros((len(sourcePointsDiagonal),waveData.get_data("SimulatedData").shape[3]))

    for ind, position in enumerate(sourcePointsDiagonal):
        newlineseries[ind] = np.mean(waveData.get_data("SimulatedData")[indices, position[0], position[1], :], axis=0)
    plt.figure()
    plt.imshow(newlineseries, aspect=4)
    plt.title(f"Channels over time (Condition {condition})")
    plt.show()

    plot = Plotting.plotfft_zoomed(np.mean(waveData.get_data("FFT_ABS")[indices,:,:],axis=0), waveData.get_sample_rate(), -20, 20, "fft abs", scale='log')
    plot.show()

    x_labels = np.arange(1)
    plt.figure()
    plt.bar(x_labels, [np.mean(result["Max Along Power"][indices])], color='b', width = 0.25 )
    plt.bar(x_labels + 0.25, [np.mean(result["Max Reverse Power"][indices])] , color='r', width = 0.25 )
    plt.legend(labels=["Along", "Reverse"])
    plt.xticks(x_labels + 0.125, ["0 degree"])
    plt.title(f"Max Power (Condition {condition})")
    plt.show()