import unittest
from unittest.mock import MagicMock
import numpy as np
from scipy import signal
from WaveSpace.Decomposition.EMD import  FreqAmpPhaseFromAnalytic, assess_harmonic_criteria, EMD
import WaveSpace.Utils.WaveData as wd

class EMD_test(unittest.TestCase):
    def setUp(self):
        # Create mock data for testing
        self.n_samples = 1000
        self.n_trials = 3
        self.n_channels = 2
        self.n_imfs = 5
        
        # Create synthetic signal (mix of sine waves)
        t = np.linspace(0, 1, self.n_samples)
        self.present_frequencies = [5, 12]  # Frequencies in Hz
        wave1 = np.sin(2 * np.pi * self.present_frequencies[0] * t) 
        wave2 = 0.5 * np.sin(2 * np.pi * self.present_frequencies[1] * t)  
        self.test_signal = wave1 + wave2
        self.test_signal = self.test_signal + np.random.normal(0, 0.1, self.n_samples)  # Add noise
        # Mock data bucket with test signal
        mock_data = np.zeros((self.n_trials, self.n_channels, self.n_samples))
        channelNames = [f"channel{ind}" for ind in range(self.n_channels)]
        for i in range(self.n_trials):
            for j in range(self.n_channels):
                mock_data[i, j, :] = self.test_signal
        
        # Create waveData object
        self.wave = wd.WaveData()
        self.testDataBucket = wd.DataBucket(mock_data, "test_bucket", "trl_chan_time", channelNames)
        self.wave.add_data_bucket(self.testDataBucket)
        self.wave.ActiveDataBucket = "test_bucket"   
        self.wave.set_sample_rate(1000)  # Set sample rate to 1000 Hz for the test     
        
    # def test_EMD_process_trial_channel(self):
    #     """Test basic EMD processing of a single channel"""
    #     pair = (0, 0)  # trial 0, channel 0
    #     currentData = self.wave.DataBuckets["test_bucket"].get_data()
        
    #     result = EMD_process_trial_channel(pair, currentData, self.n_imfs, None)
        
    #     # Check output shape and type
    #     self.assertEqual(result.shape, (self.n_imfs, self.n_samples))
    #     self.assertTrue(np.iscomplexobj(result))

    #     # Check that IMFs are ordered by frequency (highest first)
    #     freqs = np.mean(np.abs(np.diff(np.unwrap(np.angle(result), axis=1))), axis=1)
    #     self.assertTrue(np.all(np.diff(freqs) <= 1e-6))  # Allow small tolerance for numerical inaccuracies

    def test_EMD(self):
        """Test the main EMD wrapper function"""

        siftTypes = ["regular", "masked_sift", "iterated_masked_sift","ensemble_sift","multivariate_sift"]
        #for siftType in siftTypes:
        # Test with regular sift
        EMD(self.wave, nIMFs=self.n_imfs, dataBucketName="test_bucket", siftType="regular")
        
        # Verify new data bucket was added
        self.assertIn("complexData", self.wave.DataBuckets)
        result = self.wave.DataBuckets["complexData"].get_data()
        
        # Check basic output properties
        self.assertEqual(result.shape, (self.n_imfs, self.n_trials, self.n_channels, self.n_samples))
        self.assertTrue(np.iscomplexobj(result)) 


    # def test_FreqAmpPhaseFromAnalytic(self):
    #     """Test frequency, amplitude, phase extraction"""
    #     pair = (0, 0)  # trial 0, channel 0
    #     currentData = self.wave.DataBuckets["test_bucket"].get_data()
    #     analytic_signal = np.zeros((self.n_imfs, self.n_trials, self.n_channels, self.n_samples), dtype=complex)

    #     # Create mock analytic signal
    #     for trial in range(self.n_trials):
    #         for channel in range(self.n_channels):
    #             analytic_signal = EMD_process_trial_channel(pair, currentData, self.n_imfs, None)
        
    #     self.wave.DataBuckets["test_bucket"].get_data.return_value = analytic_signal

    #     IF, IA, IP = FreqAmpPhaseFromAnalytic(
    #         self.wave, 
    #         smooth_phase=5, 
    #         smooth_freq=3,
    #         dataBucketName="test_bucket"
    #     )
        
    #     # Check output shapes
    #     self.assertEqual(IF.shape, analytic_signal.shape)
    #     self.assertEqual(IA.shape, analytic_signal.shape)
    #     self.assertEqual(IP.shape, analytic_signal.shape)
        
    #     # Check frequency estimates are close to expected
    #     for ind, expected_freq in enumerate(range(self.present_frequencies)):
    #         actual_freq = np.nanmean(IF[ind, 0, 0, :])
    #         self.assertAlmostEqual(actual_freq, expected_freq, delta=0.5)
            
    def test_assess_harmonic_criteria(self):
        """Test harmonic relationship assessment"""
        # Create signals with known harmonic relationship
        t = np.linspace(0, 1, 1000)
        fundamental = np.sin(2 * np.pi * 5 * t)
        harmonic = np.sin(2 * np.pi * 10 * t + np.pi/4)
        
        # Create analytic signals
        imfs = np.column_stack([harmonic, fundamental])
        analytic_signal = signal.hilbert(imfs)
        
        # Get IF, IA, IP
        IF = np.diff(np.unwrap(np.angle(analytic_signal)), axis=0) * 1000 / (2*np.pi)
        IA = np.abs(analytic_signal)[1:]  # After diff
        IP = np.unwrap(np.angle(analytic_signal))[1:]
        
        # Test harmonic assessment
        harmonic_inds = assess_harmonic_criteria(IP, IF, IA, base_imf=1)
        
        # Should identify IMF 0 as harmonic of IMF 1
        self.assertEqual(harmonic_inds.tolist(), [0])
        

        
if __name__ == '__main__':
    unittest.main()