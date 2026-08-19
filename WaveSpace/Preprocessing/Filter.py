import mne
import numpy as np
from scipy.signal import butter, detrend, filtfilt, firwin, impulse, lfilter

from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wd


def filter_broadband(waveData,dataBucketName = "", LowCutOff=0, HighCutOff=100,  n_jobs=5):
    """Apply an MNE non-causal band-pass filter to a data bucket.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to filter.
    dataBucketName : str, default=""
        Name of the input data bucket. By default, the active data bucket is
        used.
    LowCutOff : float, default=0
        Low-frequency cutoff in Hz.
    HighCutOff : float, default=100
        High-frequency cutoff in Hz.
    n_jobs : int, default=5
        Number of parallel workers passed to MNE. (Required input for mne.filter.filter_data())

    Returns
    -------
    None
        Adds the filtered data to ``waveData`` as the ``BBFiltered`` bucket.

    Notes
    -----
    The input is temporarily reshaped to trials, channels, and time when
    necessary, then restored to its original dimensional ordering.
    """
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    desiredDimord = "trl_chan_time"
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , desiredDimord)

    NewData = mne.filter.filter_data(data = currentData,sfreq = waveData.get_sample_rate(),l_freq = LowCutOff, h_freq= HighCutOff, n_jobs=n_jobs)
    dataBucket = wd.DataBucket(NewData, "BBFiltered", desiredDimord, time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time() ,chanNames=waveData.get_channel_names())
    # reshape original data
    if hasBeenReshaped:
        dataBucket.reshape(origShape, origDimord)  

    waveData.add_data_bucket(dataBucket)    
    waveData.log_history(["Broadband Filter", "filt",LowCutOff, HighCutOff])

def filter_notch(waveData, dataBucketName = "", LineNoiseFreq = 50, n_jobs=5):
    """Apply an MNE notch filter to remove line-frequency components.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to filter.
    dataBucketName : str, default=""
        Name of the input data bucket. An empty string uses the active bucket.
    LineNoiseFreq : float or array-like, default=50
        Frequency or frequencies to remove in Hz.
    n_jobs : int, default=5
        Number of parallel workers passed to MNE.

    Returns
    -------
    None
        Adds filtered data to ``waveData`` as the ``NotchFiltered`` bucket.

    Notes
    -----
    Data are temporarily reordered to trial, channel, and time dimensions
    before filtering.
    """
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , "trl_chan_time")

    NewData = mne.filter.notch_filter(x =waveData.get_active_data(), Fs=waveData.get_sample_rate(),freqs = LineNoiseFreq, 
        filter_length = 'auto', n_jobs=n_jobs)
    dataBucket = wd.DataBucket(NewData, "NotchFiltered", waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord(),time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time() ,chanNames= waveData.get_channel_names())
    # reshape original data
    if hasBeenReshaped:
        waveData.DataBuckets[dataBucketName].reshape(origShape, origDimord)  
        waveData.add_data_bucket(dataBucket)
        # reshape last bucket
        waveData.DataBuckets[dataBucketName].reshape(origShape, origDimord)   
        waveData.log_history(["Notch Filter", "notch", LineNoiseFreq])

def bandpass(lowcut, highcut, fs, type="IIR", order=5):
    """Design IIR or FIR band-pass filter coefficients.

    Parameters
    ----------
    lowcut, highcut : float
        Lower and upper cutoff frequencies in Hz.
    fs : float
        Sampling frequency in Hz.
    type : {"IIR", "FIR"}, default="IIR"
        Filter-design method.
    order : int, default=5
        IIR filter order or FIR order before conversion to tap count.

    Returns
    -------
    b : numpy.ndarray
        Numerator filter coefficients.
    a : numpy.ndarray or list of float
        Denominator filter coefficients.
    impulse_response_length : int
        Length of the calculated impulse response.

    Notes
    -----
    FIR filters use ``order + 1`` taps and have a denominator of ``[1.0]``.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if type == "IIR":
        b, a = butter(order, [low, high], btype='band')
        # Calculate the impulse response
        _t, h = impulse((b, a))
    elif type == "FIR":
        print("CAUTION!!! Make sure your filter order is correct!\n"
            "For FIR filters, a reasonable order is about 20 times\n"
            "of what you would use for an IIR filter.\n"
            "Current order is: " + str(order) + "\n" )    
        numtaps = order + 1  # Number of taps in the FIR filter
        b = firwin(numtaps, [low, high], pass_zero=False)
        a = [1.0]  # In an FIR filter, the a coefficients are just [1.0]
        # The length of the impulse response is the number of taps
        h = b
    else:
        raise ValueError("Invalid filter type. Must be either 'IIR' or 'FIR'.")

    # Calculate the length of the impulse response
    impulse_response_length = len(h)
    
    return b, a, impulse_response_length


def filter_narrowband(waveData, dataBucketName = "", LowCutOff=0, HighCutOff=120, type= "IIR", order=5, causal=True):
    """Detrend and apply a narrowband IIR or FIR band-pass filter.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to filter.
    dataBucketName : str, default=""
        Name of the input data bucket. By default, the active data bucket is
        used.
    LowCutOff : float, default=0
        Low-frequency cutoff in Hz.
    HighCutOff : float, default=120
        High-frequency cutoff in Hz.
    type : {"IIR", "FIR"}, default="IIR"
        Filter design to use.
    order : int, default=5
        IIR filter order or FIR order before conversion to the number of taps.
    causal : bool, default=True
        If ``True``, apply a causal filter. If ``False``, apply zero-phase
        forward-backward filtering.

    Returns
    -------
    None
        Adds the filtered data to ``waveData`` as the ``NBFiltered`` bucket.

    Notes
    -----
    Data are detrended before filtering. The function prints the impulse
    response length of the selected filter.
    """
    # ensure proper bookkeeping of data dimensions
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , "trl_chan_time")

    # Detrend the data
    currentData = detrend(currentData)

    b, a, impulse_response_length = bandpass(LowCutOff, HighCutOff, waveData.get_sample_rate(),type=type, order=order)
    print("CAUTION!!! Impulse response length: " + str(impulse_response_length))
    
    # Apply the filter.
    NewData = lfilter(b, a, currentData) if causal else filtfilt(b, a, currentData)

    if hasBeenReshaped:
        NewData = np.reshape(NewData, origShape)

    dataBucket = wd.DataBucket(NewData, "NBFiltered", waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord(),time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time() ,chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)    
    waveData.log_history(["Narrowband Filter", "filt", LowCutOff, HighCutOff, "Type: " + type, "Causal: " + str(causal)])