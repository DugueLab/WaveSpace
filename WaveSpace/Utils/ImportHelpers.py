import scipy.io as io 
import pickle

def load_MNE_data(filename):
    """Load epoched MNE data from disk.

    Parameters
    ----------
    filename : str or path-like
        Path to an MNE epochs file supported by :func:`mne.read_epochs`.

    Returns
    -------
    mne.Epochs
        Loaded epoched MNE data.

    Notes
    -----
    The MNE dependency is imported only when this function is called.
    """
    import mne
    data = mne.read_epochs(filename)
    return data

def load_MNE_fif_data(filename):
    """Load continuous MNE FIF data from disk.

    Parameters
    ----------
    filename : str or path-like
        Path to a raw FIF file.

    Returns
    -------
    mne.io.BaseRaw
        Loaded raw MNE recording with samples preloaded into memory.

    Notes
    -----
    The MNE dependency is imported only when this function is called, and
    ``preload=True`` is passed to :func:`mne.io.read_raw_fif`.
    """
    import mne.io
    data = mne.io.read_raw_fif(filename, preload=True)
    return data

def load_channel_positions(filename):
    """Load channel coordinates from a serialized position file.

    Parameters
    ----------
    filename : str or path-like
        Path to a pickle file containing the expected channel-position
        structure.

    Returns
    -------
    numpy.ndarray
        The first three coordinates of every contact except the final entry in
        the fifth object stored in the serialized structure.

    Notes
    -----
    This loader expects a project-specific pickle layout and may not work for
    arbitrary channel-position files. Pickle files may execute arbitrary code
    when loaded; only use trusted files.
    """
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
    """Serialize a WaveData object to a pickle file.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object to serialize.
    filename : str or path-like
        Destination file path.

    Returns
    -------
    None
        The object is written using the highest available pickle protocol.

    Notes
    -----
    Files produced by this function should be loaded only from trusted sources
    because pickle deserialization can execute arbitrary code.
    """
    f = open(filename, 'wb')
    pickle.dump(waveData, f, pickle.HIGHEST_PROTOCOL)
    f.close()

def load_mat_file(filename):
    """Load variables from a MATLAB MAT file.

    Parameters
    ----------
    filename : str or path-like
        Path to a MAT file readable by :func:`scipy.io.loadmat`.

    Returns
    -------
    dict
        Mapping of MATLAB variable names to loaded Python values.
    """
    return io.loadmat(filename)