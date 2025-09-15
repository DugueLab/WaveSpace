
Sensor Layout
=============
Sets of functions to transform data between 3d and 2d coordinates. Most **WaveSpace** analysis pipelines require sensor descriptions on a regular grid.
EEG, MEG sensors usually come with 3d coordinates. 

.. contents:: Table of Contents    


Setup
-----

Add the project root directory to the Python path when working with source code, not necessary when package is installed

.. code-block:: python

    import sys
    import os
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, path )
    print(path)

    from WaveSpace.Utils import ImportHelpers
    from WaveSpace.SpatialArrangement import SensorLayout as sensors
    from WaveSpace.PlottingHelpers import Plotting
    from WaveSpace.Utils import WaveData as wd

    import vtk
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle


Loading Data
------------

.. code-block:: python

    #load data from file:
    saveFolder = "ExampleData/Output/"
    waveData = ImportHelpers.load_wavedata_object(saveFolder + "SimulatedData")


Regular Grid and Distance Matrix
--------------------------------

.. code-block:: python

    #regular grid
    #calculate sensor to sensor distance, where chanpos has x and y coordinates
    sensors.regularGrid(waveData) #adds a distance matrix to the data object 

    #plot
    plt.imshow(waveData.get_distMat(), origin= 'lower')
    plt.colorbar()
    plt.title('Contact-to-Contact distance')
    plt.xlabel('Contact')
    plt.ylabel('Contact')


Distance Matrix for Regular Grid of Contacts
--------------------------------------------

.. code-block:: python

    #project 3D coordinates to 2D space, preserving distanes between them as good as possible
    sensors.distmat_to_2d_coordinates_MDS(waveData)
    #plot 3D and 2D contact positions:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(waveData.get_channel_positions()[:,0], waveData.get_channel_positions()[:,1],waveData.get_channel_positions()[:,2])
    plt.title('Contact position 3D ')
    plt.figure()
    plt.scatter(waveData.get_2d_coordinates()[:,0], waveData.get_2d_coordinates()[:,1])
    plt.title('Contact position 2D embedding preserving inter-contact distances. Arbitrary units')


Assigning Spatial Layout to Channels
------------------------------------

.. code-block:: python

    # pick some channels and assign a spatial layout to them (this makes no sense at all for real data and
    #is only to demonstrate the interpolation to a reagular grid from 3d positions)
    gridshape = waveData.get_data('SimulatedData').shape[1:3]
    chanpos = np.load( saveFolder + 'exampleChanpos.npy')
    waveData.set_channel_positions(chanpos)

    x_normalized = (chanpos[:, 0] - np.min(chanpos[:, 0])) / (np.max(chanpos[:, 0]) - np.min(chanpos[:, 0]))
    y_normalized = (chanpos[:, 1] - np.min(chanpos[:, 1])) / (np.max(chanpos[:, 1]) - np.min(chanpos[:, 1]))
    z_normalized = (chanpos[:, 2] - np.min(chanpos[:, 2])) / (np.max(chanpos[:, 2]) - np.min(chanpos[:, 2]))

    x_indices = np.round(x_normalized * (gridshape[0] - 1)).astype(int)
    y_indices = np.round(y_normalized * (gridshape[1] - 1)).astype(int)
    z_indices = np.round(z_normalized * (gridshape[1] - 1)).astype(int)

    original_data = waveData.get_data('SimulatedData')
    grid_data = np.repeat(original_data[:, :, :, np.newaxis, :], gridshape[1], axis=3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_indices, y_indices, z_indices)
    X, Y, Z = np.meshgrid(np.arange(gridshape[0]), np.arange(gridshape[1]), np.arange(gridshape[1]))
    ax.scatter(X, Y, Z, alpha=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
    #now we simply subsample the original grid at the channel positions
    newdata = grid_data[:, y_indices, x_indices, z_indices, :]
    dataBucket = wd.DataBucket(newdata, "EEGLayout",'trl_chan_time', [] )
    waveData.add_data_bucket(dataBucket) 


Interpolation and Surface Mapping
---------------------------------

.. code-block:: python

    #interpolate the whole thing back into a regular grid 
    chanInds=True
    surface, polySurface = sensors.create_surface_from_points(waveData,
                                                                type='channels',
                                                                num_points=1000)

    sensors.distance_along_surface(waveData, surface, tolerance=0.1, get_extent = chanInds, plotting= True)
    sensors.distmat_to_2d_coordinates_Isomap(waveData) #can also use MDS here

    grid_x, grid_y, mask =sensors.interpolate_pos_to_grid(
        waveData, 
        dataBucketName = "EEGLayout",
        numGridBins=18,
        return_mask = True,
        mask_stretching = True)

    distMat = sensors.regularGrid(waveData)

    original_data_bucket = "EEGLayout"
    interpolated_data_bucket = "EEGLayoutInterpolated"
    OrigInd  = (0,slice(None),202)
    InterpInd =(0,slice(None),slice(None),202)
    fig = Plotting.plot_interpolated_data(waveData, original_data_bucket, interpolated_data_bucket,
                                            grid_x, grid_y, OrigInd, InterpInd,  type='')
