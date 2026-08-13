import scipy.io as io 
import pickle

def load_MNE_data(filename):
    import mne
    data = mne.read_epochs(filename)
    return data

def load_MNE_fif_data(filename):
    import mne.io
    data = mne.io.read_raw_fif(filename, preload=True)
    return data

def load_channel_positions(filename):
    with open(filename,'rb') as f:
        ChannelPositions = pickle.load(f)
    #% load contact positions and surfaces
    chanPos = ChannelPositions[4][:-1,0:3]
    #[KP] Fix this
    return chanPos

def load_wavedata_object(filename):
    """Load a serialized :class:`WaveData` object from a pickle file.

    Parameters
    ----------
    filename : str or path-like
        Path to a file created with :func:`save_wavedata_object`.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        The deserialized WaveData object.

    Notes
    -----
    Pickle files may execute arbitrary code when loaded. Only load files from
    trusted sources.
    """
    with open(filename,'rb') as f:
        waveData = pickle.load(f)
    return waveData

def save_wavedata_object(waveData, filename):
    f = open(filename, 'wb')
    pickle.dump(waveData, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load_mat_file(filename):
    
    return io.loadmat(filename)