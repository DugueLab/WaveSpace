import mne
import numpy as np
from scipy.signal import butter, detrend, filtfilt, firwin, impulse, lfilter

from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wd


def filter_broadband(waveData,dataBucketName = "", LowCutOff=0, HighCutOff=100,  n_jobs=5):
    import mne
    '''wrapper for MNE non-causal filter'''
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
    '''wrapper forMNE non-causal filter'''
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
    '''Design a bandpass filter using either an IIR or FIR approach. Returns filter coefficients and impulse response length.'''
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
    '''Scipy zero-phase bandpass filter. Detrends before narrowband filtering.'''
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
    
    # Apply the filter
    if causal:
        # For a causal filter, use lfilter
        NewData = lfilter(b, a, currentData)
    else:
        # For a non-causal filter, use filtfilt
        NewData = filtfilt(b, a, currentData)

    if hasBeenReshaped:
        NewData = np.reshape(NewData, origShape)

    dataBucket = wd.DataBucket(NewData, "NBFiltered", waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord(),time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time() ,chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)    
    waveData.log_history(["Narrowband Filter", "filt", LowCutOff, HighCutOff, "Type: " + type, "Causal: " + str(causal)])