"""
Circular-linear correlation. Uses a Python implementation based on https://github.com/mullerlab/generalized-phase.git
Requires complex data in a regular grid with known (relative) distances between grid points. 

"""
#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import DistanceCorrelation
from WaveSpace.SpatialArrangement import SensorLayout as sensors

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib import colormaps
# Load some simulated data
dataPath  = os.path.join(path, "Examples/ExampleData/Output") 
waveData = ImportHelpers.load_wavedata_object(dataPath + "/complexData")

#%% 
# we already know that our data is on a regular grid because we generated it that way
# so we can simply use the channel positions to create a distance matrix

sensors.regularGrid(waveData)

#%% Generalized phase distance correlation
DistanceCorrelation.calculate_distance_correlation_GP(waveData, dataBucketName = "complexData", evaluationAngle=np.pi, tolerance=0.2)
dataFrame = waveData.get_data("PhaseDistanceCorrelation")

#%%
#Calculate distance correlation based on selected sourcepoints
pointRange = range(0,20,2)
sourcePoints = []
for i in pointRange:
    sourcePoints.append((i,i))
DistanceCorrelation.calculate_distance_correlation(waveData, dataBucketName = "complexData", sourcePoints=sourcePoints, pixelSpacing=1)

#%% Plot phase-distance correlation over time for selected points on diagonal
phaseDistCorr= waveData.get_data("PhaseDistanceCorrelation")
shape = waveData.get_data("complexData").shape
dimord = waveData.DataBuckets["complexData"].get_dimord()
splitDimord = dimord.split("_")
spatialIndexStart = splitDimord.index("posx")
selectedTrial = 0
fig, ax = plt.subplots(figsize=(8,6))
for i, point in enumerate(sourcePoints):
    rho = phaseDistCorr.loc[(phaseDistCorr["trialInd"] == selectedTrial) & (phaseDistCorr["sourcePointX"] == point[0]) & (phaseDistCorr["sourcePointY"] == point[1])]
    color = Plotting.getProbeColor(i, len(sourcePoints))
    ax.plot(rho["rho"].tolist(), label =str(point), color=color)
ax.legend()
color_grid = Plotting.get_color_grid_from_probes((shape[spatialIndexStart],shape[spatialIndexStart+1]), sourcePoints)
Plotting.add_color_grid_legend(ax, color_grid, position=[0.2, 0.2, 1.5, 1.5])
plt.show()

# %% Calculate and plot average phase-distance correlation for 600 to 1000 ms for all points
pointRange = (20,20)
sourcePoints = []
for x in range(pointRange[0]):
    for y in range(pointRange[1]):
        sourcePoints.append((x,y))

DistanceCorrelation.calculate_distance_correlation(waveData, dataBucketName = "complexData", sourcePoints=sourcePoints, pixelSpacing=1)

output_path = os.path.join(path, "Examples/ExampleData/Output")
waveData.save_to_file(os.path.join(output_path, "DistanceCorrelation"))


#%%
waveData = ImportHelpers.load_wavedata_object("ExampleData/Output/DistanceCorrelation")
pointRange = (20,20)
sourcePoints = []
for x in range(pointRange[0]):
    for y in range(pointRange[1]):
        sourcePoints.append((x,y))

phaseDistCorr= waveData.get_data("PhaseDistanceCorrelation")
conditions = waveData.get_trialInfo()[::2]

shape = waveData.get_data("complexData").shape
selectedTrial = 4

rho = np.zeros((8,20,20))
for condInd, condition in enumerate(conditions):
    for i, (x,y) in enumerate(sourcePoints):
        phaseDistCorrOverTime = phaseDistCorr.loc[(phaseDistCorr["trialInd"] == condInd*2) &
                                                (phaseDistCorr["sourcePointX"] == x) &
                                                (phaseDistCorr["sourcePointY"] == y)]
        rho[condInd,x,y] = np.mean(phaseDistCorrOverTime["rho"][300:500])
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(rho[condInd], origin="lower", )
    ax.set_title(condition)
    fig.colorbar(im, ax=ax)
    plt.show()


