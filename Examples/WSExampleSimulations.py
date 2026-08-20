"""
Simulations
===========
Simulating and Visualizing Traveling Waves with WaveSpace

This tutorial demonstrates how to use the **WaveSpace** toolbox to simulate
different spatiotemporal wave patterns, add noise, and visualize the results.
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

from WaveSpace.PlottingHelpers import Plotting
from WaveSpace.Simulation import SimulationFuns

# %%
# Simulation Conditions
# ---------------------
# We will generate several types of waves:
#
# * Plane waves
# * Target waves (inward/outward)
# * Rotating waves (spirals)
# * Local oscillations (random vs. synchronized)
Conditions = ["PlaneWave_45", "PlaneWave_135", "TargetWave_in", "TargetWave_out", "RotatingWave_CW", "RotatingWave_CCW", "LocalOscillationRandom", "LocalOscillationSynched"]

# %%
# Plane Wave Simulation
# ---------------------
# We start with unidirectional plane waves mixed with spatial noise.
Type = "PlaneWave"  # PlaneWave	 StationaryPulse   TargetWave	RotatingWave	LocalOscillation	SpatialPinkNoise	WhiteNoise
nTrials = 4
MatrixSize = 20
SampleRate = 500
SimDuration = 1.6

SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
TemporalFrequency = [10, 10, 10, 10]
WaveDirection = [45, 45, 135, 135]
SimLayout = "grid"  # Grid, Radial, Circular

# These options only apply after mixing the wave with noise,
# for now they will just return a mask to be used later
WaveOnset = [300, 300, 300, 300]  # Onset in ms
WaveDuration = 1000  # Duration in ms, note:
# After waveduration has passed, the current cycle of the wave will finish

planeWave = SimulationFuns.simulate_signal(
    Type,
    nTrials,
    MatrixSize,
    SampleRate,
    SimDuration,
    SimLayout=SimLayout,
    # SimOptions from here on
    TemporalFrequency=TemporalFrequency,
    SpatialFrequency=SpatialFrequency,
    WaveDirection=WaveDirection,
    WaveOnset=WaveOnset,
    WaveDuration=WaveDuration,
    )

planeWaveNoise = SimulationFuns.simulate_signal(
        Type="SpatialPinkNoise",
        ntrials=nTrials,
        MatrixSize=MatrixSize,
        SampleRate=SampleRate,
        SimLayout=SimLayout,
        SimDuration=SimDuration)

SNR = 0.8
planeWaveData = SimulationFuns.SNRMix(planeWave, planeWaveNoise, SNR, SimLayout="grid")

# %%
# Target Wave Simulation
# ----------------------
Type = "TargetWave"
nTrials = 4
matrixSize = 20
SampleRate = 500
SimDuration = 1.6

CenterX = 2
CenterY = 2

SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
TemporalFrequency = [10, 10, 10, 10]

# lower or higher than 0 determines in or outward motion for targetWave
WaveDirection = [-1, -1, 1, 1]

# initialize data
WaveOnset = 300
WaveDuration = 1000

targetWave = SimulationFuns.simulate_signal(
    Type,
    nTrials,
    matrixSize,
    SampleRate,
    SimDuration,
    SimLayout="grid",
    # SimOptions from here on
    TemporalFrequency=TemporalFrequency,
    SpatialFrequency=SpatialFrequency,
    WaveDirection=WaveDirection,
    WaveOnset=WaveOnset,
    WaveDuration=WaveDuration,
    CenterX=CenterX,
    CenterY=CenterY
    )
SNR = 0.8
targetNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
targetWaveData = SimulationFuns.SNRMix(targetWave, targetNoise, SNR, SimLayout="grid")

# %%
# Rotating Wave Simulation
# ------------------------
Type = "RotatingWave"
nTrials = 4
matrixSize = 20
SampleRate = 500
SimDuration = 1.6

CenterX = 2
CenterY = 2

SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
TemporalFrequency = [10, 10, 10, 10]

# lower or higher than 0 determines rotating clockwise or counter clockwise for spiral wave
WaveDirection = [1, 1, -1, -1]

# initialize data
WaveOnset = 300
WaveDuration = 1000

spiralWave = SimulationFuns.simulate_signal(
    Type,
    nTrials,
    matrixSize,
    SampleRate,
    SimDuration,
    SimLayout="grid",
    # SimOptions from here on
    TemporalFrequency=TemporalFrequency,
    SpatialFrequency=SpatialFrequency,
    WaveDirection=WaveDirection,
    WaveOnset=WaveOnset,
    WaveDuration=WaveDuration,
    CenterX=CenterX,
    CenterY=CenterY
    )

SNR = 0.8
spiralNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
spiralWaveData = SimulationFuns.SNRMix(spiralWave, spiralNoise, SNR, SimLayout="grid")

# %%
# Local Oscillation Simulation
# ----------------------------
Type = "LocalOscillation"
nTrials = 4
matrixSize = 20
SampleRate = 500
SimDuration = 1.6

CenterX = 2
CenterY = 2

SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
TemporalFrequency = [10, 10, 10, 10]

# lower or higher than 0 determines rotating clockwise or counter clockwise for spiral wave
WaveDirection = [1, 1, -1, -1]

# Oscillators can have random phase relative to each other, or be synchronized
OscillatoryPhase = ["Random", "Random", "Synched", "Synched"]  # Random, Synched

# initialize data
WaveOnset = 300
WaveDuration = 1000

# Create Oscillators
localOscillators = SimulationFuns.simulate_signal(
        Type=Type,
        ntrials=nTrials,
        MatrixSize=MatrixSize,
        SampleRate=SampleRate,
        SimDuration=SimDuration,
        SimLayout="grid",

        WaveOnset=WaveOnset,
        WaveDuration=WaveDuration,
        OscillatoryPhase="Random",
        TemporalFrequency=TemporalFrequency,
        OscillatorProportion=0.4
    )

SNR = 0.8
oscillatorNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, matrixSize, SampleRate, SimDuration, SimLayout="grid")
oscillatorWaveData = SimulationFuns.SNRMix(localOscillators, oscillatorNoise, SNR, SimLayout="grid")

# %%
# Combining Data
# --------------
# We now combine all simulated data into one dataset.
simCondList = []

for item in Conditions:
    simCondList.append(item)
    simCondList.append(item)

waveData = SimulationFuns.combine_SimData([planeWaveData, targetWaveData, spiralWaveData, oscillatorWaveData], dimension='trl', SimCondList=simCondList)
# save for later use
_data_env = os.environ.get("WAVEDATA_EXAMPLE_DIR")
data_dir = Path(_data_env) if _data_env else Path(__file__).parent / "ExampleData"
output_dir = data_dir / "Output"
if not os.environ.get("BUILDING_DOCS"):
    output_dir.mkdir(parents=True, exist_ok=True)
    waveData.save_to_file(output_dir / "SimulatedData.pkl")

# %%
# Visualization
# -------------
# We can animate the simulated wave activity across the grid.
if not os.environ.get("BUILDING_DOCS"):
    for trl in range(waveData.get_data("SimulatedData").shape[0]):
        ani = Plotting.animate_grid_data(waveData, DataBucketName="SimulatedData", dataInd=trl, probepositions=[(0, 15), (5, 15), (10, 15), (15, 15), (19, 15), (19, 15)])
        ani.save(output_dir / f"SimulationAnimation_{trl}.gif")

# %%
# Next Steps
# ----------
# * Explore other wave types such as **StationaryPulse** or **WhiteNoise**.
# * Adjust **SNR**, **frequencies**, and **onsets** to simulate different experimental conditions.
# * Use plotting helpers to create static or interactive visualizations of your simulated data.
