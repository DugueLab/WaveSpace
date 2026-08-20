
******************
WaveSpace
******************
Python tools for the simulation and analysis of cortical traveling waves

.. contents:: Table of Contents
  :depth: 3

Statement of Need
=================

Cortical traveling waves are spatiotemporal patterns of neural activity propagating across the cortical surface over time. They have been observed in a variety of neural recordings and have been associated with a range of cognitive processes including perception, memory, and attention. Analyzing these patterns requires methods that account for both the temporal structure of neural signals and their spatial organization across recording sensors or cortical locations.

**WaveSpace** is a Python toolbox designed to support the simulation, detection and analysis of cortical traveling waves.
WaveSpace is intended primarily for **researchers and scientists working with electrophysiological and other spatiotemporal neural data**, including users working with EEG, MEG, and simulated neural recordings. It is designed to provide a common workflow for researchers who need to investigate how neural activity varies across both space and time.

The package also aims to facilitate reproducible research by providing a structured representation of neural data, analysis functions, simulation tools, and automated tests that can be used to verify the software's functionality.


Installation
============

WaveSpace requires Python 3.9 or newer. The package is tested with Python 3.9 through Python 3.14.

Install from PyPI
-----------------

The recommended way to install WaveSpace is with `pip`:

.. code-block:: console

  $ pip install wavespace


This installs WaveSpace together with its required Python dependencies.

To install a specific version of WaveSpace, for example version 1.1.8:

.. code-block:: console

    $ pip install wavespace==1.1.8


Installing a specific version is recommended when reproducing analyses from a scientific publication, as it ensures that the same software version is used.

Dependencies
------------

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
------------------------

To contribute to WaveSpace or work with the source code, clone the repository and install the development dependencies:

.. code-block:: console

   $ git clone https://github.com/kpetras/WaveSpace.git
   $ cd WaveSpace
   $ uv sync --group dev

The development environment includes the tools required to run the test suite and test WaveSpace across its supported Python versions.

For more information about contributing and testing, see the `Contributing guide <https://github.com/kpetras/WaveSpace/blob/main/CONTRIBUTING.md>`_.

Examples
========

WaveSpace includes a set of tutorials demonstrating common workflows for
simulation and analysis of cortical traveling waves.

.. toctree::
   :maxdepth: 2

   auto_examples/index

The WaveData Class
------------------

The ``WaveData`` class serves as a container for time-series data related to cortical traveling waves. It provides functionalities for data storage, manipulation, and analysis, ensuring a structured workflow for handling multi-channel neural recordings.

Key Features
~~~~~~~~~~~~

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


API Reference
=============

.. toctree::
   :maxdepth: 2

   source/api

Testing
=======

WaveSpace includes an automated test suite based on Python's built-in
``unittest`` framework. The tests are located in the ``UnitTest`` directory.

To run the tests locally from the repository root:

.. code-block:: bash

   python -m unittest discover -s UnitTest -p "*_test.py"

The test suite can also be run against all supported Python versions using
``tox``:

.. code-block:: bash

   tox

The project uses GitHub Actions to automatically run the test suite for
Python 3.9 through 3.14 on pushes to the ``main`` branch and on pull
requests. This helps ensure that changes remain compatible with the
supported Python versions.


Community and Support
---------------------

WaveSpace welcomes contributions from researchers and developers interested
in improving the package.

Contributing
------------

Contributions should follow the guidelines described in the
``CONTRIBUTING.md`` file in the GitHub repository. Contributors are encouraged
to create a separate branch for their changes, add or update tests where
appropriate, and submit a pull request to the ``main`` branch.

Reporting issues
----------------

If you encounter a bug or unexpected behaviour, please report it by opening
an issue on the
`WaveSpace GitHub issue tracker <https://github.com/kpetras/WaveSpace/issues>`_.
Please include enough information to reproduce the problem, including the
WaveSpace version, Python version, operating system, and a minimal example
where possible.

Feature requests and questions
------------------------------

Feature requests and questions can also be submitted through the GitHub issue
tracker. Using public issues allows other users and contributors to benefit
from previous discussions and solutions.
