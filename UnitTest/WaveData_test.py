import unittest
from unittest.mock import MagicMock
import numpy as np
import WaveSpace.Utils.WaveData as wd

class wavedata_test(unittest.TestCase):
    def setUp(self):
        print("done")
        self.nTrials = 3
        self.nChannels = 16
        self.nTimepoints = 500
        self.mock_data = np.zeros((self.nTrials,self.nChannels,self.nTimepoints))        
        self.waveData = wd.WaveData()
        self.channelNames = [f"channel_{i}" for i in range(self.nChannels)]
        testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time",self.channelNames)
        self.waveData.add_data_bucket(testDataBucket)
        self.waveData.ActiveDataBucket = "test_bucket"   
        self.waveData.set_sample_rate(1000)

    def test_add_dataBucket(self):       
        newTestDataBucket = wd.DataBucket(self.mock_data, "test_new_bucket", "trl_chan_time", self.channelNames)
        self.waveData.add_data_bucket(newTestDataBucket)
        self.waveData.ActiveDataBucket = "test_new_bucket"   
        self.assertIsNotNone(self.waveData.DataBuckets["test_new_bucket"], "Basic DataBucket not created")
        self.assertIsNotNone(self.waveData.DataBuckets["test_bucket"], "Original databucket is gone")

    def test_existence_of_time(self):
        #Tests whether time vector exists and have right shape, for default databucket created on setup
        databuckettime = self.waveData.DataBuckets["test_bucket"].get_time()
        self.assertIsNotNone(databuckettime, "time vector for databucket does not exist")
        databucketshape = self.waveData.get_data("test_bucket").shape
        self.assertEqual(databucketshape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")
    
    def test_existence_of_time_on_waveData(self):
        #Tests whether time vector exists for waveData Object
        databuckettime = self.waveData.get_time()
        self.assertIsNotNone(databuckettime, "time vector for databucket does not exist")
        databucketshape = self.waveData.get_data("test_bucket").shape
        self.assertEqual(databucketshape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")

    def test_setting_of_time_when_creating_dataBucket(self):
       # No timevector supplied
       testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time", self.channelNames)
       self.assertIsNotNone(testDataBucket.get_time())
       self.assertEqual(self.mock_data.shape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")

    def test_setting_of_time_when_creating_dataBucket_and_samplerate(self):
       # No timevector supplied
       testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time",sampleRate=500,chanNames=self.channelNames)
       self.assertIsNotNone(testDataBucket.get_time())
       self.assertEqual(self.mock_data.shape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")

    def test_setting_of_time_when_creating_dataBucket_and_timevec(self):
       # adding timevector without samplerate
       timeVector = np.arange(0,self.mock_data.shape[2],1)
       testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time", time=timeVector, chanNames= self.channelNames)
       self.assertIsNotNone(testDataBucket.get_time())
       self.assertEqual(self.mock_data.shape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")

    def test_setting_of_time_when_creating_dataBucket_and_timevec_and_samplerate(self):
       # adding timevector without samplerate
       timeVector = np.arange(0,self.mock_data.shape[2],1)
       testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time", time=timeVector, sampleRate=500, chanNames= self.channelNames)
       self.assertIsNotNone(testDataBucket.get_time())
       self.assertEqual(self.mock_data.shape[2], len(self.waveData.DataBuckets["test_bucket"].get_time()), "length of time vector does not match length of data")
    
    def test_catch_on_illegal_timevector(self):
        try:
            timeVector = np.arange(0,20,1)
            testDataBucket = wd.DataBucket(self.mock_data, "test_bucket", "trl_chan_time", time=timeVector, sampleRate=500, chanNames= self.channelNames)
            self.assertTrue(False, "databucket does not catch illegal timevectors")
        except:
            self.assertTrue(True)