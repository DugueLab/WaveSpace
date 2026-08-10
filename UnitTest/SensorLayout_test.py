import os
import unittest
import numpy as np
import WaveSpace.Utils.WaveData as wd
import WaveSpace.SpatialArrangement.SensorLayout as sensors

TESTDATA_DIR = os.path.join(os.path.dirname(__file__), 'TestData')

class SensorLayout_test(unittest.TestCase):
    def setUp(self):
        self.nTrials = 3
        self.nChannels = 74
        self.nTimepoints = 500
        self.mock_data = np.zeros((self.nTrials,self.nChannels,self.nTimepoints))  
        self.waveData = wd.WaveData()
        chanpos = np.load(os.path.join(TESTDATA_DIR, 'exampleChanpos.npy'))
        self.waveData.set_channel_positions(chanpos)

        # create values between 0 and 1 based on y distance
        max_y = np.max(chanpos[:, 1])
        min_y = np.min(chanpos[:, 1])
        # Normalize y-values to range [0, 1]
        normalized_y = (chanpos[:, 1] - min_y) / (max_y - min_y)
        channel_values = normalized_y.reshape(1, self.nChannels, 1)
        self.mock_data[:] = channel_values        

        self.channelNames = [f"channel_{i}" for i in range(self.nChannels)]
        testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time",self.channelNames)
        self.waveData.add_data_bucket(testDataBucket)
        self.waveData.ActiveDataBucket = "test_bucket"   
        self.waveData.set_sample_rate(1000)


    def test_Interpolation_from_positions(self):
        chanInds=True
        numGridBins=18        
        surface, polySurface = sensors.create_surface_from_points(self.waveData,
                                                                    type='channels',
                                                                    num_points=1000)

        sensors.distance_along_surface(self.waveData, surface, tolerance=0.1, get_extent = chanInds, plotting= False)
        sensors.distmat_to_2d_coordinates_Isomap(self.waveData) #can also use MDS here

        grid_x, grid_y, mask =sensors.interpolate_pos_to_grid(
            self.waveData, 
            dataBucketName = "test_bucket",
            numGridBins=numGridBins,
            return_mask = True,
            mask_stretching = True)

        testData = self.waveData.get_data("test_bucketInterpolated")

        # test if shape of output is correct
        self.assertEqual(testData.shape, (self.nTrials, numGridBins, numGridBins, self.nTimepoints) ,"Output data has wrong shape")
        
        # Values should increase along the y-axis in the middle column
        middleColumn = testData[0,:,9,0]
        self.assertFalse(np.any([y < y_prime for y, y_prime in zip(testData, testData[1:])]))
