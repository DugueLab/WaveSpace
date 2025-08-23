#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.Utils import WaveData as wd
import numpy as np

#%%

# Placeholder data
nTrials = 3
nChannels = 64
nTimepoints = 500

data = np.random.rand(nTrials, nChannels, nTimepoints)  # 10 time points, 20x20 grid
 
# Obligatory: Create time-vector
duration = 1 # in seconds
start = -0.5
time = np.linspace(start, start+duration, nTimepoints)  

# Obligatory: Sample rate
sampleRate = nTimepoints / duration

#Obligatory: dimension order
dimord = "trl_chan_time"  # trials x channels x time

#Obligatory: channel Names
channelNames = [f"Channel {i+1}" for i in range(nChannels)]

#optional: unit
unit = "uV"

#optional (but highly recommended): channel Positions
chanpos = np.random.rand(nChannels, 3)  # Random 3D positions for each channel

#optional: trial info
trialInfo = ['condition_1', 'condition_2', 'condition_3']

#optional: distance matrix (can also be calculated from positions)
distMat   = np.random.rand(nChannels, nChannels)


#create empty waveData object
waveData = wd.WaveData()
waveData.set_sample_rate(sampleRate)
#Create Databucket: 
dataBucketName = "fakeEEG"
fakeEEG = wd.DataBucket(data,dataBucketName, dimord= dimord, chanNames=channelNames, time = time, unit=unit)
waveData.add_data_bucket(fakeEEG)

waveData.set_channel_positions(chanpos)
waveData.set_trialInfo(trialInfo)
waveData.set_distMat(distMat)

