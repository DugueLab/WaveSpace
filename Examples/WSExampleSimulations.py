#%%
import sys
import os

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.Simulation import SimulationFuns
from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Utils import HelperFuns as hf

import numpy as np
import matplotlib.pyplot as plt

Conditions = ["PlaneWave_45", "PlaneWave_135", "TargetWave_in", "TargetWave_out", "RotatingWave_CW", "RotatingWave_CCW", "LocalOscillationRandom", "LocalOscillationSynched"]

#%% Single Unidirectional waves and some noise, high SNR
Type =  "PlaneWave" # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 4
MatrixSize = 20
SampleRate= 500
SimDuration= 1.6

SpatialFrequency = [0.6,0.6,0.6,0.6]
TemporalFrequency = [10,10,10,10]
WaveDirection = [45,45,135,135]  
SimLayout = "grid" # Grid, Radial, Circular

# These options only apply after mixing the wave with noise, for now they will just return a mask to be used later
WaveOnset = [300,300,300,300] # Onset in ms
WaveDuration = 1000 # Duration in ms, note: 
#After waveduration has passed, the current cycle of the wave will finish

planeWave = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    MatrixSize, 
    SampleRate, 
    SimDuration, 
    SimLayout = SimLayout,
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration,
    )

planeWaveNoise = SimulationFuns.simulate_signal(
        Type="SpatialPinkNoise", 
        ntrials = nTrials, 
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate,
        SimLayout = SimLayout,
        SimDuration= SimDuration)

SNR = 0.8
planeWaveData = SimulationFuns.SNRMix(planeWave, planeWaveNoise, SNR, SimLayout="grid")

#%% Add Target Waves

Type = "TargetWave"
nTrials = 4
matrixSize = 20
SampleRate= 500
SimDuration= 1.6

CenterX = 2    
CenterY = 2

SpatialFrequency = [0.6,0.6,0.6,0.6]
TemporalFrequency = [10, 10, 10, 10] 

#lower or higher than 0 determines in or outward motion for targetWave
WaveDirection = [-1, -1, 1, 1]  

#initialize data
WaveOnset =  300
WaveDuration = 1000

targetWave = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    SimLayout = "grid",
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration, 
    CenterX = CenterX,
    CenterY = CenterY
    )
SNR = 0.8
targetNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
targetWaveData = SimulationFuns.SNRMix(targetWave, targetNoise, SNR, SimLayout="grid")

#%% Add Spiral wave
Type = "RotatingWave"
nTrials = 4
matrixSize = 20
SampleRate= 500
SimDuration= 1.6

CenterX = 2    
CenterY = 2

SpatialFrequency = [0.6,0.6,0.6,0.6]
TemporalFrequency = [10, 10, 10, 10] 

#lower or higher than 0 determines rotating clockwise or counter clockwise for spiral wave
WaveDirection = [1, 1, -1, -1]  

#initialize data
WaveOnset =  300
WaveDuration = 1000

spiralWave = SimulationFuns.simulate_signal(
    Type, 
    nTrials, 
    matrixSize, 
    SampleRate, 
    SimDuration, 
    SimLayout = "grid",
    #SimOptions from here on
    TemporalFrequency = TemporalFrequency,
    SpatialFrequency = SpatialFrequency,
    WaveDirection = WaveDirection,
    WaveOnset = WaveOnset,
    WaveDuration = WaveDuration, 
    CenterX = CenterX,
    CenterY = CenterY
    )

SNR = 0.8
spiralNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
spiralWaveData = SimulationFuns.SNRMix(spiralWave, spiralNoise, SNR, SimLayout="grid")

#%% Local Oscillation noise
Type="LocalOscillation"
nTrials = 4
matrixSize = 20
SampleRate= 500
SimDuration= 1.6

CenterX = 2    
CenterY = 2

SpatialFrequency = [0.6,0.6,0.6,0.6]
TemporalFrequency = [10, 10, 10, 10] 

#lower or higher than 0 determines rotating clockwise or counter clockwise for spiral wave
WaveDirection = [1, 1, -1, -1]  

#Oscillators can have random phase relative to each other, or be synchronized
OscillatoryPhase = ["Random","Random","Synched","Synched"]  # Random, Synched

#initialize data
WaveOnset =  300
WaveDuration = 1000

#Create Oscillators
localOscillators = SimulationFuns.simulate_signal(
        Type=Type, 
        ntrials = nTrials,        
        MatrixSize = MatrixSize, 
        SampleRate= SampleRate, 
        SimDuration= SimDuration, 
        SimLayout = "grid",

        WaveOnset = WaveOnset,
        WaveDuration = WaveDuration,
        OscillatoryPhase = "Random",
        TemporalFrequency = TemporalFrequency,
        OscillatorProportion = 0.4
    )

SNR = 0.8
oscillatorNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
oscillatorWaveData = SimulationFuns.SNRMix(localOscillators, oscillatorNoise, SNR, SimLayout="grid")

#%%
simCondList = []

for item in Conditions:
    simCondList.append(item)
    simCondList.append(item)

waveData = SimulationFuns.combine_SimData([planeWaveData, targetWaveData, spiralWaveData, oscillatorWaveData], dimension = 'trl', SimCondList = simCondList)
# save for later use
output_path = "ExampleData/Output"
if not os.path.exists(output_path):
    os.makedirs(output_path)
waveData.save_to_file(os.path.join(output_path, "SimulatedData"))

#%% Plot an example timeseries (takes long)
for trl in range(waveData.get_data("SimulatedData").shape[0]):
    ani = Plotting.animate_grid_data(waveData, DataBucketName="SimulatedData", dataInd=trl, probepositions=[(0,15), (5,15), (10,15), (15,15), (19,15), (19,15)])
    plot_file = output_path + f"/SimulationAnimation_{trl}.gif"
    ani.save(plot_file)




