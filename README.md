# WaveSpace version 1.1.7
WaveSpace is a Python toolbox for simulating, detecting, and analyzing spatiotemporal traveling waves in neural sensor array data. It provides  tools for generating synthetic datasets, and applying a range of wave analysis techniques such as optical flow, 2D FFT, circular-linear correlation and singular value decomposition. In addition, it contains pipelines to decompose multi-dimensional timeseries data into its frequency components to derive robust phase estimates. WaveSpace’s WaveData class provides a structured approach to managing complex datasets, while its plotting helpers facilitate intuitive visualization of spatiotemporal patterns.

## Documentation
Access latest documentation from [here](https://wavespace.readthedocs.io/en/latest/)

## Installation

Download latest version from [here](https://github.com/kpetras/WaveSpace/tree/main/package/dist)

Open a terminal, navigate to the directory you downloaded to and install with
```
pip install WaveSpace-1.1.7-py3-none-any.whl
```

## Testing
   * Run tests locally using Python's built-in `unittest` framework from the `UnitTest` folder:

     ```bash
     python -m unittest discover UnitTest

## Contributing
See https://github.com/kpetras/WaveSpace/blob/main/CONTRIBUTING.md

## Modules
### Decomposition: 
Implements various frequency decomposition techniques, such as Fourier and wavelet transforms, Empirical Mode Decomposition (EMD)

### Preprocessing:
Provides functions for cleaning, normalizing, and filtering time series data.

### Plotting Helpers: 
Contains utilities for visualizing cortical traveling waves using matplotlib & pyvista, including time-frequency plots, phase maps, and spatial-temporal representations.

### Simulation: 
Tools for generating synthetic cortical traveling waves, aiding in model validation and hypothesis testing.

### Spatial Arrangement:
Handles spatial organization of sensor positions. Includes interpolation options

### Statistics:
Offers methods for computing null distributions.

### Utils:
A collection of general-purpose helper functions used throughout the toolbox, including data manipulation and file I/O.
#### The `WaveData` Class

The `WaveData` class serves as a container for time-series data related to cortical traveling waves. It provides functionalities for data storage, manipulation, and analysis, ensuring a structured workflow for handling multi-channel neural recordings. 

##### **Key Features**
- **Initialization (`__init__`)**: Stores channel positions, time vectors, sample rates, and maintains a structured dataset with multiple *DataBuckets* for flexible data handling.
- **Data Management**:
  - Supports multiple datasets through *DataBuckets*, enabling users to store, retrieve, and manipulate data flexibly.
  - Provides methods to add, delete, and check the existence of specific *DataBuckets*.
  - Allows appending datasets and setting an active dataset for streamlined analysis.
- **Data Processing**:
  - **Cropping**: Enables temporal cropping of data using specific time intervals.
  - **Trial Pruning**: Removes unwanted trials from datasets while maintaining metadata consistency.
- **Metadata Handling**:
  - Stores and retrieves spatial arrangements of recording channels (`set_channel_positions`, `get_channel_positions`).
  - Maintains a history of operations for reproducibility (`log_history`).
  - Supports storage and retrieval of simulation and trial metadata (`set_simInfo`, `get_trialInfo`).
- **I/O and Persistence**:
  - Saves objects to files for later retrieval (`save_to_file`).
  - Provides a structured string representation (`__repr__`) for quick dataset summaries.

This class is essential for organizing and processing large-scale neural recordings, offering flexibility in data structuring, preprocessing, and visualization. Let me know if you’d like any refinements!

### Wave Analysis: 
Core module for detecting, characterizing, and quantifying cortical traveling waves using advanced signal processing techniques.

![overview](JOSS/WaveSpace_overview_smaller.png)
