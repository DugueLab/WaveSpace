#%%

# Add the project root directory to the Python path when working with source code, 
# not necessary when package is installed
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, path )
print(path)

from WaveSpace.Utils import ImportHelpers
from WaveSpace.SpatialArrangement import SensorLayout as sensors
from WaveSpace.PlottingHelpers import Plotting

import vtk
import numpy as np
import matplotlib.pyplot as plt
import pickle
#%%
#load data from file:
saveFolder = "ExampleData/Output/"
waveData = ImportHelpers.load_wavedata_object(saveFolder + "SimulatedData")

#%%
#regular grid
#calculate sensor to sensor distance, where chanpos has x and y coordinates
sensors.regularGrid(waveData) #adds a distance matrix to the data object 

#plot
plt.imshow(waveData.get_distMat(), origin= 'lower')
plt.colorbar()
plt.title('Contact-to-Contact distance')
plt.xlabel('Contact')
plt.ylabel('Contact')


#%% Distance Matrix for regular grid of contacts
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

#%%
# pick some channels and assign a spatial layout to them (this makes no sense at all for real data and
    #is only to demonstrate the interpolation to a reagular grid from 3d positions)
chanpos = np.load(saveFolder + 'exampleChanpos.npy')
waveData.set_channel_positions(chanpos)


chanInds=True
surface, polySurface = sensors.create_surface_from_points(waveData,
                                                            type='channels',
                                                            num_points=1000)

sensors.distance_along_surface(waveData, surface, tolerance=0.1, get_extent = chanInds, plotting= True)
sensors.distmat_to_2d_coordinates_Isomap(waveData) #can also use MDS here
# grid_x, grid_y, mask =sensors.interpolate_pos_to_grid(
#     waveData, 
#     numGridBins=18,
#     return_mask=True, 
#     mask_stretching=True
#     )
grid_x, grid_y, mask =sensors.interpolate_pos_to_grid(
    waveData, 
    dataBucketName = "SimulatedData",
    numGridBins=18,
    return_mask = True,
    mask_stretching = True)

distMat = sensors.regularGrid(waveData)

#%% cortical distance
#[KP]Provide downsampled samplesurface + sample positions
SurfaceFile = './ExampleData/surfaceFileInflated_LH' #path to freesurfer generated cortical surface
with open(SurfaceFile, 'rb') as f:
    Surface = pickle.load(f)

sensors.distance_along_surface(waveData,Surface)

plt.imshow(waveData.get_distMat(), origin= 'lower')
plt.colorbar()
plt.title('Contact-to-Contact distance along surface in m')
plt.xlabel('SeedContact')
plt.ylabel('TargetContact')

# %%