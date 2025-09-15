
Simulations
===========
Simulating and Visualizing Traveling Waves with WaveSpace

This tutorial demonstrates how to use the **WaveSpace** toolbox to simulate different spatiotemporal wave patterns, add noise, and visualize the results.

.. contents:: Table of Contents    


Setup
-----

Before running the examples, ensure that the project root is on the Python path:

.. code-block:: python

    import sys
    import os

    # Add project root to Python path (only necessary if not installed as a package)
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, path)
    print(path)

    from WaveSpace.Simulation import SimulationFuns
    from WaveSpace.PlottingHelpers import Plotting
    from WaveSpace.Utils import HelperFuns as hf

    import numpy as np
    import matplotlib.pyplot as plt


Simulation Conditions
---------------------

We will generate several types of waves:

* Plane waves
* Target waves (inward/outward)
* Rotating waves (spirals)
* Local oscillations (random vs. synchronized)

.. code-block:: python


    Conditions = [
      "PlaneWave_45", "PlaneWave_135",
      "TargetWave_in", "TargetWave_out",
      "RotatingWave_CW", "RotatingWave_CCW",
      "LocalOscillationRandom", "LocalOscillationSynched"
    ]


Plane Wave Simulation
---------------------

We start with unidirectional plane waves mixed with spatial noise.

.. code-block:: python

    Type = "PlaneWave"
    nTrials = 4
    MatrixSize = 20
    SampleRate = 500
    SimDuration = 1.6

    SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
    TemporalFrequency = [10, 10, 10, 10]
    WaveDirection = [45, 45, 135, 135]
    SimLayout = "grid"

    WaveOnset = [300, 300, 300, 300]
    WaveDuration = 1000

    planeWave = SimulationFuns.simulate_signal(
        Type, nTrials, MatrixSize, SampleRate, SimDuration,
        SimLayout=SimLayout,
        TemporalFrequency=TemporalFrequency,
        SpatialFrequency=SpatialFrequency,
        WaveDirection=WaveDirection,
        WaveOnset=WaveOnset,
        WaveDuration=WaveDuration
    )

    planeWaveNoise = SimulationFuns.simulate_signal(
        Type="SpatialPinkNoise",
        ntrials=nTrials,
        MatrixSize=MatrixSize,
        SampleRate=SampleRate,
        SimLayout=SimLayout,
        SimDuration=SimDuration
    )

    SNR = 0.8
    planeWaveData = SimulationFuns.SNRMix(planeWave, planeWaveNoise, SNR, SimLayout="grid")


Target Wave Simulation
----------------------

.. code-block:: python

    Type = "TargetWave"
    CenterX, CenterY = 2, 2

    SpatialFrequency = [0.6, 0.6, 0.6, 0.6]
    TemporalFrequency = [10, 10, 10, 10]
    WaveDirection = [-1, -1, 1, 1]  # -1 = inward, 1 = outward

    WaveOnset = 300
    WaveDuration = 1000

    targetWave = SimulationFuns.simulate_signal(
        Type, nTrials, MatrixSize, SampleRate, SimDuration,
        SimLayout="grid",
        TemporalFrequency=TemporalFrequency,
        SpatialFrequency=SpatialFrequency,
        WaveDirection=WaveDirection,
        WaveOnset=WaveOnset,
        WaveDuration=WaveDuration,
        CenterX=CenterX,
        CenterY=CenterY
    )

    targetNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, MatrixSize, SampleRate, SimDuration, SimLayout="grid")
    targetWaveData = SimulationFuns.SNRMix(targetWave, targetNoise, 0.8, SimLayout="grid")
    

Rotating Wave Simulation
------------------------

.. code-block:: python

    Type = "RotatingWave"
    WaveDirection = [1, 1, -1, -1]  # 1 = CW, -1 = CCW

    spiralWave = SimulationFuns.simulate_signal(
        Type, nTrials, MatrixSize, SampleRate, SimDuration,
        SimLayout="grid",
        TemporalFrequency=TemporalFrequency,
        SpatialFrequency=SpatialFrequency,
        WaveDirection=WaveDirection,
        WaveOnset=WaveOnset,
        WaveDuration=WaveDuration,
        CenterX=CenterX,
        CenterY=CenterY
    )

    spiralNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, MatrixSize, SampleRate, SimDuration, SimLayout="grid")
    spiralWaveData = SimulationFuns.SNRMix(spiralWave, spiralNoise, 0.8, SimLayout="grid")


Local Oscillation Simulation
----------------------------

.. code-block:: python

    Type = "LocalOscillation"

    localOscillators = SimulationFuns.simulate_signal(
        Type=Type, ntrials=nTrials,
        MatrixSize=MatrixSize, SampleRate=SampleRate, SimDuration=SimDuration,
        SimLayout="grid",
        WaveOnset=WaveOnset,
        WaveDuration=WaveDuration,
        OscillatoryPhase="Random",
        TemporalFrequency=TemporalFrequency,
        OscillatorProportion=0.4
    )

    oscillatorNoise = SimulationFuns.simulate_signal("SpatialPinkNoise", nTrials, MatrixSize, SampleRate, SimDuration, SimLayout="grid")
    oscillatorWaveData = SimulationFuns.SNRMix(localOscillators, oscillatorNoise, 0.8, SimLayout="grid")


Combining Data
--------------

We now combine all simulated data into one dataset.

.. code-block:: python

    simCondList = []
    for item in Conditions:
        simCondList.append(item)
        simCondList.append(item)

    waveData = SimulationFuns.combine_SimData(
        [planeWaveData, targetWaveData, spiralWaveData, oscillatorWaveData],
        dimension='trl',
        SimCondList=simCondList
    )

    output_path = "ExampleData/Output"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    waveData.save_to_file(os.path.join(output_path, "SimulatedData"))


Visualization
-------------

We can animate the simulated wave activity across the grid.

.. code-block:: python

    for trl in range(waveData.get_data("SimulatedData").shape[0]):
        ani = Plotting.animate_grid_data(
            waveData,
            DataBucketName="SimulatedData",
            dataInd=trl,
            probepositions=[(0,15), (5,15), (10,15), (15,15), (19,15), (19,15)]
        )
        plot_file = output_path + f"/SimulationAnimation_{trl}.gif"
        ani.save(plot_file)

Next Steps
----------

* Explore other wave types such as **StationaryPulse** or **WhiteNoise**.
* Adjust **SNR**, **frequencies**, and **onsets** to simulate different experimental conditions.
* Use plotting helpers to create static or interactive visualizations of your simulated data.
