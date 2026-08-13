"""
Creating a WaveData Object
===========================
This tutorial demonstrates how to create a `WaveData` object in **WaveSpace**
from scratch, including adding data, metadata, and optional information such
as channel positions and trial info.
"""

# %%
# Setup
# -----
import numpy as np

from WaveSpace.Utils import WaveData

# %%
# Creating Placeholder Data
# -------------------------
# Placeholder data
nTrials = 3
nChannels = 64
nTimepoints = 500

data = np.random.rand(nTrials, nChannels, nTimepoints)  # 3 trials, 64 channels, 500 time points

# Obligatory: Create time-vector
duration = 1  # in seconds
start = -0.5
time = np.linspace(start, start + duration, nTimepoints)

# Obligatory: Sample rate
sampleRate = nTimepoints / duration

# Obligatory: dimension order
dimord = "trl_chan_time"  # trials x channels x time

# Obligatory: channel Names
channelNames = [f"Channel {i + 1}" for i in range(nChannels)]

# optional: unit
unit = "uV"

# optional (but highly recommended): channel Positions
chanpos = np.random.rand(nChannels, 3)  # Random 3D positions for each channel

# optional: trial info
trialInfo = ['condition_1', 'condition_2', 'condition_3']

# optional: distance matrix (can also be calculated from positions)
distMat = np.random.rand(nChannels, nChannels)

# %%
# Creating and Populating the WaveData Object
# -------------------------------------------
# create empty waveData object
waveData = WaveData.WaveData()
waveData.set_sample_rate(sampleRate)
# Create Databucket:
dataBucketName = "fakeEEG"
fakeEEG = WaveData.DataBucket(data, dataBucketName, dimord=dimord, chanNames=channelNames, time=time, unit=unit)
waveData.add_data_bucket(fakeEEG)

waveData.set_channel_positions(chanpos)
waveData.set_trialInfo(trialInfo)
waveData.set_distMat(distMat)
