import unittest
import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

# Mocking posxy_to_chan if it's from another module
def posxy_to_chan(data):
    shape = data.shape
    new_shape = shape[:-3] + (shape[-3] * shape[-2], shape[-1])
    reshaped = data.reshape(new_shape)
    return reshaped, (shape[-3], shape[-2])

# Include the function under test here (force_dimord)
# Or import it from your module

class helperfuns_test(unittest.TestCase):

    def test_force_dimord_no_change(self):
        data = np.random.randn(2, 3, 4)
        changed, new_data = hf.force_dimord(data, "cond_freq_time", "cond_freq_time")
        self.assertFalse(changed)
        self.assertTrue(np.array_equal(data, new_data))

    def test_force_dimord_posxy_to_chan(self):
        data = np.random.randn(2, 5, 6, 10)  # trl, posx, posy, time
        changed, new_data = hf.force_dimord(data, "trl_posx_posy_time", "trl_chan_time")
        self.assertTrue(changed)
        self.assertEqual(new_data.shape, (2, 30, 10))  # 5x6=30

    def test_force_dimord_complex_case(self):
        data = np.random.randn(2, 3, 4, 5, 100)  # cond, freq, trl, chan, time
        changed, new_data = hf.force_dimord(data, "cond_freq_trl_chan_time", "trl_chan_time")
        self.assertTrue(changed)
        self.assertEqual(new_data.shape, (24, 5, 100))  # 2x3x4 = 24 trials

    def test_force_dimord_invalid_expand_posxy(self):
        data = np.random.randn(10, 20, 30)
        with self.assertRaises(Exception):
            hf.force_dimord(data, "chan_time", "posx_posy_time")

    def test_force_dimord_insert_trl_to_posxy(self):
        data = np.random.randn(5, 6, 100)  # posx, posy, time
        changed, new_data = hf.force_dimord(data, "posx_posy_time", "trl_posx_posy_time")
        self.assertTrue(changed)
        self.assertEqual(new_data.shape, (1, 5, 6, 100))