"""
Wave Basis Functions
====================
This tutorial demonstrates how to explore spatial wave activity using the
WaveSpace toolbox. We will import wave data, extract spatial basis
functions, and visualize them.
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

from WaveSpace.Utils import ImportHelpers
from WaveSpace.WaveAnalysis import WaveActivity

# %%
# Load Data
# ---------
# Load an example wave data object from file
_data_env = os.environ.get("WAVEDATA_EXAMPLE_DIR")
data_dir = Path(_data_env) if _data_env else Path(__file__).parent / "ExampleData"
waveData = ImportHelpers.load_wavedata_object(data_dir / "complexData.pkl")

# %%
# Exploring Basis Functions for a Subset
# ---------------------------------------
# We can look at the spatial basis functions for a subset of trials
# (belonging to the same condition) and at a particular frequency of interest.
# Here, we specify the indices into the data bucket:
nBases = 3
dataInd = (slice(0, 1), slice(10, 12), slice(None), slice(None), slice(None))
WaveActivity.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

bases = waveData.get_data('Bases')

fig, axs = plt.subplots(1, nBases, figsize=(nBases * 6, 6))
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
    axs[b].set_title(f'wave map {b + 1}')
    axs[b].set_xlabel('posy')
    axs[b].set_ylabel('posx')
    fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

plt.tight_layout()
plt.show()

# %%
# Extracting Bases from All Data
# ------------------------------
# Alternatively, we can calculate the bases on **all data at once** and then
# sort out the weights later. This provides a more global view of wave activity.
nBases = 5
dataInd = None
WaveActivity.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

bases = waveData.get_data('Bases')

fig, axs = plt.subplots(1, nBases, figsize=(nBases * 6, 6))
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
    axs[b].set_title(f'wave map {b + 1}')
    axs[b].set_xlabel('posy')
    axs[b].set_ylabel('posx')
    fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

plt.tight_layout()
plt.show()

# %%
# Interpreting the Results
# -------------------------
# The bases have changed to express linear combinations of all the waves
# we put in. When using the full dataset, the bases represent **linear
# combinations** of all included waves, rather than just those from a
# restricted subset. This allows for more general spatial modes but may
# mix different underlying dynamics.
