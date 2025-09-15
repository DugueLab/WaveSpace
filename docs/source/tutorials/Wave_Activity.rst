Wave basis functions
=======================

This tutorial demonstrates how to explore spatial wave activity using
the `WaveSpace` toolbox. We will import wave data, extract spatial
basis functions, and visualize them.

.. contents:: Table of Contents

Setup
-----

First, add the project root directory to the Python path when working
with source code (not necessary if `WaveSpace` is already installed as
a package):

.. code-block:: python

    import sys
    import os
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, path )
    print(path)


Import Required Packages
------------------------

We import the relevant modules from `WaveSpace` along with some
utilities for data handling and visualization:

.. code-block:: python

    from WaveSpace.WaveAnalysis import WaveActivity as wa
    from WaveSpace.Utils import ImportHelpers
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import cm


Load Data
---------

Load an example wave data object from file:

.. code-block:: python

    waveData = ImportHelpers.load_wavedata_object("ExampleData/Output/complexData")


Exploring Basis Functions for a Subset
--------------------------------------

We can look at the spatial basis functions for a subset of trials
(belonging to the same condition) and at a particular frequency of
interest. Here, we specify the indices into the data bucket:

.. code-block:: python

    nBases = 3
    dataInd = (slice(0,1), slice(10,12), slice(None), slice(None), slice(None))
    wa.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

    bases = waveData.get_data('Bases')


Now, visualize the phase maps of the extracted bases:

.. code-block:: python

    fig, axs = plt.subplots(1, nBases, figsize=(nBases*6, 6))
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
        axs[b].set_title(f'wave map {b+1}')
        axs[b].set_xlabel('posy')
        axs[b].set_ylabel('posx')
        fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

    plt.tight_layout()
    plt.show()


This produces a series of phase maps that represent the spatial basis
functions extracted from the subset of trials.

Extracting Bases from All Data
------------------------------

Alternatively, we can calculate the bases on **all data at once** and
then sort out the weights later. This provides a more global view of
wave activity.

.. code-block:: python

    nBases = 5
    dataInd = None
    wa.find_wave_activity(waveData, dataBucketName="complexData", dataInd=dataInd, nBases=nBases)

    bases = waveData.get_data('Bases')

    fig, axs = plt.subplots(1, nBases, figsize=(nBases*6, 6))
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
        axs[b].set_title(f'wave map {b+1}')
        axs[b].set_xlabel('posy')
        axs[b].set_ylabel('posx')
        fig.colorbar(im, ax=axs[b], fraction=0.046, pad=0.04, label='Phase (rad)')

    plt.tight_layout()
    plt.show()


Interpreting the Results
------------------------

When using the full dataset, the bases represent **linear combinations**
of all included waves, rather than just those from a restricted subset.
This allows for more general spatial modes but may mix different
underlying dynamics.
