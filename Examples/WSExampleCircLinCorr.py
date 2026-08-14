"""
Circular-Linear Correlation
===========================
This tutorial demonstrates how to use the **WaveSpace** toolbox to compute
circular-linear (phase-distance) correlations on simulated data, including
grid setup, distance matrix calculation, and visualization of results.

Uses a Python implementation based on
https://github.com/mullerlab/generalized-phase.git
Requires complex data in a regular grid with known (relative) distances
between grid points.
"""

# %%
# Setup
# -----
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.SpatialArrangement import SensorLayout
from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import DistanceCorrelation

# %%
# Loading Simulated Data
# ----------------------
# Load some simulated data
_data_env = os.environ.get("WAVEDATA_EXAMPLE_DIR")
data_dir = Path(_data_env) if _data_env else Path(__file__).parent / "ExampleData"
output_dir = data_dir / "Output"
waveData = ImportHelpers.load_wavedata_object(data_dir / "complexData.pkl")

# %%
# Grid and Distance Matrix Setup
# ------------------------------
# We already know that our data is on a regular grid because we generated it
# that way, so we can simply use the channel positions to create a distance matrix.
SensorLayout.regularGrid(waveData)

# %%
# Generalized Phase Distance Correlation
# --------------------------------------
DistanceCorrelation.calculate_distance_correlation_GP(waveData, dataBucketName="complexData", evaluationAngle=np.pi, tolerance=0.2)
dataFrame = waveData.get_data("PhaseDistanceCorrelation")

# %%
# Distance Correlation for Selected Source Points
# ------------------------------------------------
# Calculate distance correlation based on selected sourcepoints
pointRange = range(0, 20, 2)
sourcePoints = []
for i in pointRange:
    sourcePoints.append((i, i))
DistanceCorrelation.calculate_distance_correlation(waveData, dataBucketName="complexData", sourcePoints=sourcePoints, pixelSpacing=1)

# %%
# Plotting Phase-Distance Correlation Over Time
# ----------------------------------------------
phaseDistCorr = waveData.get_data("PhaseDistanceCorrelation")
shape = waveData.get_data("complexData").shape
dimord = waveData.DataBuckets["complexData"].get_dimord()
splitDimord = dimord.split("_")
spatialIndexStart = splitDimord.index("posx")
selectedTrial = 0
fig, ax = plt.subplots(figsize=(8, 6))
for i, point in enumerate(sourcePoints):
    rho = phaseDistCorr.loc[(phaseDistCorr["trialInd"] == selectedTrial) & (phaseDistCorr["sourcePointX"] == point[0]) & (phaseDistCorr["sourcePointY"] == point[1])]
    color = Plotting.getProbeColor(i, len(sourcePoints))
    ax.plot(rho["rho"].tolist(), label=str(point), color=color)
ax.legend()
color_grid = Plotting.get_color_grid_from_probes((shape[spatialIndexStart], shape[spatialIndexStart + 1]), sourcePoints)
Plotting.add_color_grid_legend(ax, color_grid, position=[0.2, 0.2, 1.5, 1.5])
plt.show()

# %%
# Full Grid Correlation and Visualization
# ----------------------------------------
# Calculate and plot average phase-distance correlation for 600 to 1000 ms
# for all points (only do if you have too much time on your hands)
if not os.environ.get("BUILDING_DOCS"):
    pointRange = (20, 20)
    sourcePoints = []
    for x in range(pointRange[0]):
        for y in range(pointRange[1]):
            sourcePoints.append((x, y))

    DistanceCorrelation.calculate_distance_correlation(waveData, dataBucketName="complexData", sourcePoints=sourcePoints, pixelSpacing=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    waveData.save_to_file(output_dir / "DistanceCorrelation.pkl")

# %%
# Loading and Plotting Saved Correlation Data
# -------------------------------------------
if not os.environ.get("BUILDING_DOCS"):
    waveData = ImportHelpers.load_wavedata_object(output_dir / "DistanceCorrelation.pkl")
    pointRange = (20, 20)
    sourcePoints = []
    for x in range(pointRange[0]):
        for y in range(pointRange[1]):
            sourcePoints.append((x, y))

    phaseDistCorr = waveData.get_data("PhaseDistanceCorrelation")
    conditions = waveData.get_trialInfo()[::2]

    shape = waveData.get_data("complexData").shape
    selectedTrial = 4

    rho = np.zeros((8, 20, 20))
    for condInd, condition in enumerate(conditions):
        for i, (x, y) in enumerate(sourcePoints):
            phaseDistCorrOverTime = phaseDistCorr.loc[(phaseDistCorr["trialInd"] == condInd * 2) &
                                                    (phaseDistCorr["sourcePointX"] == x) &
                                                    (phaseDistCorr["sourcePointY"] == y)]
            rho[condInd, x, y] = np.mean(phaseDistCorrOverTime["rho"][300:500])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(rho[condInd], origin="lower")
        ax.set_title(condition)
        fig.colorbar(im, ax=ax)
        plt.show()
