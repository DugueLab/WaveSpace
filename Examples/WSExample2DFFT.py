"""
2D FFT Analysis
===============
This tutorial demonstrates how to use the **WaveSpace** toolbox to perform 2D
FFT analysis on simulated data, including selecting source points, running
the FFT, and visualizing the results.
"""

# %%
# Setup
# -----
# Optional: Add the project root directory to the Python path if you have just checked out the repository and
# did not install the package into your Python environment:
#
# .. code-block:: python
#
#    import os
#    import sys
#    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#    sys.path.insert(0, path)
#
# Otherwise just import

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import WaveAnalysis

# %%
# Loading Simulated Data
# ----------------------
_data_env = os.environ.get("WAVEDATA_EXAMPLE_DIR")
data_dir = Path(_data_env) if _data_env else Path(__file__).parent / "ExampleData"
waveData = ImportHelpers.load_wavedata_object(data_dir / "SimulatedData.pkl")

# %%
# Selecting Source Points
# -----------------------
# Create 10 sample points along the diagonal of the channel array
gridSize = waveData.get_data("SimulatedData").shape[1]
nPoints = range(0, gridSize, 2)
sourcePointsDiagonal = []
for i in nPoints:
    sourcePointsDiagonal.append([i, i])

# %%
# Running 2D FFT Analysis
# -----------------------
# Restrict to (temp) frequencies between lower and upper bound:
lowerBound = 2
upperBound = 40

WaveAnalysis.FFT_2D(waveData, sourcePointsDiagonal, lowerBound, upperBound, DataBucketName="SimulatedData")

result = waveData.get_data("Result")

# %%
# Visualizing Results
# -------------------
n_trials = waveData.get_data("SimulatedData").shape[0]
trialInfo = waveData.get_trialInfo()
conditions = np.unique(trialInfo)

for condition in conditions:
    indices = [i for i, cond in enumerate(trialInfo) if cond == condition]
    logRatios = np.mean(np.log(result["Max Along Power"][indices] / result["Max Reverse Power"][indices]))
    newlineseries = np.zeros((len(sourcePointsDiagonal), waveData.get_data("SimulatedData").shape[3]))

    for ind, position in enumerate(sourcePointsDiagonal):
        newlineseries[ind] = np.mean(waveData.get_data("SimulatedData")[indices, position[0], position[1], :], axis=0)
    plt.figure()
    plt.imshow(newlineseries, aspect=4)
    plt.title(f"Channels over time (Condition {condition})")
    plt.show()

    plot = Plotting.plotfft_zoomed(np.mean(waveData.get_data("FFT_ABS")[indices, :, :], axis=0), waveData.get_sample_rate(), -20, 20, "fft abs", scale='log')
    plot.show()

    x_labels = np.arange(1)
    plt.figure()
    plt.bar(x_labels, [np.mean(result["Max Along Power"][indices])], color='b', width=0.25)
    plt.bar(x_labels + 0.25, [np.mean(result["Max Reverse Power"][indices])], color='r', width=0.25)
    plt.legend(labels=["Along", "Reverse"])
    plt.xticks(x_labels + 0.125, ["0 degree"])
    plt.title(f"Max Power (Condition {condition})")
    plt.show()
