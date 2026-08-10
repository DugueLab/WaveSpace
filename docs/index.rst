
******************
WaveSpace
******************
Python tools for the simulation and analysis of cortical traveling waves

.. contents:: Table of Contents

Statement of Need
-----------------

Cortical traveling waves are spatiotemporal patterns of neural activity in which patterns of activity propagate across the cortical surface over time. They have been observed in a variety of neural recordings and have been associated with processes including perception, cognition, and large-scale brain dynamics. Analyzing these patterns requires methods that account for both the temporal structure of neural signals and their spatial organization across recording sensors or cortical locations.

**WaveSpace** is a Python toolbox designed to support the simulation, detection, characterization, and analysis of cortical traveling waves. It provides tools for working with multichannel neural recordings, including preprocessing and frequency decomposition, spatial interpolation and sensor organization, wave-specific analyses, statistical procedures, visualization, and the generation of simulated traveling-wave data.

WaveSpace is intended primarily for **researchers and scientists working with electrophysiological and other spatiotemporal neural data**, including users working with EEG, MEG, and simulated neural recordings. It is designed to provide a common workflow for researchers who need to investigate how neural activity varies across both space and time.

The package also aims to facilitate reproducible research by providing a structured representation of neural data, analysis functions, simulation tools, and automated tests that can be used to verify the software's functionality.


Installation
------------

WaveSpace requires Python 3.9 or newer. The package is tested with Python 3.9 through Python 3.14.

Install from PyPI
~~~~~~~~~~~~~~~~~

The recommended way to install WaveSpace is with `pip`:

.. code-block:: console
  
  $ pip install WaveSpace


This installs WaveSpace together with its required Python dependencies.

To install a specific version of WaveSpace, for example version 1.1.8:

.. code-block:: console

    $ pip install WaveSpace==1.1.8


Installing a specific version is recommended when reproducing analyses from a scientific publication, as it ensures that the same software version is used.

Dependencies
~~~~~~~~~~~~

WaveSpace automatically installs the dependencies required for its core functionality. These currently include:

* NumPy
* Matplotlib
* SciPy
* Plotly
* Pint
* PyVista
* pandas
* scikit-learn
* scikit-image
* tvb-gdist
* EMD
* MNE

The required dependencies are specified in the package metadata and are installed automatically by `pip`.

Development installation
~~~~~~~~~~~~~~~~~~~~~~~~

To contribute to WaveSpace or work with the source code, clone the repository and install the development dependencies:

.. code-block:: console

   $ git clone https://github.com/kpetras/WaveSpace.git
   $ cd WaveSpace
   $ uv sync --group dev

The development environment includes the tools required to run the test suite and test WaveSpace across its supported Python versions.

For more information about contributing and testing, see the [Contributing guide](https://github.com/kpetras/WaveSpace/blob/main/CONTRIBUTING.md).

Examples
--------

WaveSpace includes a set of tutorials demonstrating common workflows for simulation and analysis of cortical traveling waves. The examples progress from creating and manipulating `WaveData` objects through signal decomposition, spatial organization, and wave-specific analyses.

The tutorials include:

* [Creating a WaveData Object](https://wavespace.readthedocs.io/en/latest/source/tutorials/Create_WaveData.html)
* [Simulating WaveData](https://wavespace.readthedocs.io/en/latest/source/tutorials/Simulate_WaveData.html)
* [Frequency Decomposition](https://wavespace.readthedocs.io/en/latest/source/tutorials/Frequency_Decomposition.html)
* [Sensor Layout](https://wavespace.readthedocs.io/en/latest/source/tutorials/Sensor_layout.html)
* [2D FFT Analysis](https://wavespace.readthedocs.io/en/latest/source/tutorials/2DFFT.html)
* [Circular-Linear Correlation](https://wavespace.readthedocs.io/en/latest/source/tutorials/CircLinCorr.html)
* [Wave Basis Functions](https://wavespace.readthedocs.io/en/latest/source/tutorials/Wave_Activity.html)
* [Optical Flow Analysis](https://wavespace.readthedocs.io/en/latest/source/tutorials/optical_flow.html)

These examples are intended both as introductions to the package and as starting points for applying WaveSpace to real or simulated neural data.

API Reference
=============

.. toctree::
   :maxdepth: 2

   source/api

******************
Modules
******************

Decomposition
-------------
Implements various frequency decomposition techniques, such as Fourier and wavelet transforms, Empirical Mode Decomposition (EMD).

Preprocessing
-------------
Provides functions for cleaning, normalizing, and filtering time series data.

Plotting Helpers
----------------
Contains utilities for visualizing cortical traveling waves using matplotlib & pyvista, including time-frequency plots, phase maps, and spatial-temporal representations.

Simulation
----------
Tools for generating synthetic cortical traveling waves, aiding in model validation and hypothesis testing.

Spatial Arrangement
-------------------
Handles spatial organization of sensor positions. Includes interpolation options.

Statistics
----------
Offers methods for computing null distributions.

Utils
-----
A collection of general-purpose helper functions used throughout the toolbox, including data manipulation and file I/O.

Wave Analysis
-------------
Core module for detecting, characterizing, and quantifying cortical traveling waves using advanced signal processing techniques.

******************
The WaveData Class
******************

The ``WaveData`` class serves as a container for time-series data related to cortical traveling waves. It provides functionalities for data storage, manipulation, and analysis, ensuring a structured workflow for handling multi-channel neural recordings.

Key Features
------------

- **Initialization (`__init__`)**: Stores channel positions, time vectors, sample rates, and maintains a structured dataset with multiple *DataBuckets* for flexible data handling.
- **Data Management**:
  - Supports multiple datasets through *DataBuckets*, enabling users to store, retrieve, and manipulate data flexibly.
  - Provides methods to add, delete, and check the existence of specific *DataBuckets*.
  - Allows appending datasets and setting an active dataset for streamlined analysis.
- **Data Processing**:
  - Cropping: Enables temporal cropping of data using specific time intervals.
  - Trial Pruning: Removes unwanted trials from datasets while maintaining metadata consistency.
- **Metadata Handling**:
  - Stores and retrieves spatial arrangements of recording channels (``set_channel_positions``, ``get_channel_positions``).
  - Maintains a history of operations for reproducibility (``log_history``).
  - Supports storage and retrieval of simulation and trial metadata (``set_simInfo``, ``get_trialInfo``).
- **I/O and Persistence**:
  - Saves objects to files for later retrieval (``save_to_file``).
  - Provides a structured string representation (``__repr__``) for quick dataset summaries.

This class is essential for organizing and processing large-scale neural recordings, offering flexibility in data structuring, preprocessing, and visualization. 
