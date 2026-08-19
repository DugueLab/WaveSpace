import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

from scipy.optimize import curve_fit
from numpy.lib.stride_tricks import sliding_window_view

def wavelet_convolution(waveData, frequencies, n_cycles=3, dataBucketName=None):
    """Compute a tapered-Gaussian Morlet transform in the time domain.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the time series to transform.
    frequencies : array-like of float
        Center frequencies of the wavelets.
    n_cycles : int, default=3
        Number of cycles in each Morlet wavelet.
    dataBucketName : str, default=None
        Name of the input data bucket. By default, the active data bucket is
        used.

    Returns
    -------
    None
        Adds complex wavelet coefficients to ``waveData`` as the
        ``complexData`` bucket with a leading ``freq`` dimension.

    Notes
    -----
    Edge samples are removed according to the lowest requested frequency, so
    the output time axis is shorter than the input time axis.

    References
    ----------
    https://doi.org/10.1371/journal.pone.0148413
    https://doi.org/10.1371/journal.pcbi.1007316
    """
    if not dataBucketName:
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
        
    hf.assure_consistency(waveData)    
    data = waveData.get_data(dataBucketName)
    oldshape = data.shape
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    desiredDimord = "trl_chan_time"
    hasBeenReshaped, data =  hf.force_dimord(data, currentDimord , desiredDimord)

    nTrials, nChans, nTime = data.shape
    frequencies = np.asarray(frequencies)
    n_freqs = len(frequencies)

    dt_ms = 1000/waveData.get_sample_rate()
    pad_min = int(n_cycles * 500.0 / (np.min(frequencies) * dt_ms))
    output_len = nTime - 2 * pad_min

    if output_len < 1:
        raise ValueError(f"Time series is too short for the selected frequency and number of cycles. \n " \
        f"Requested {n_cycles} at {np.min(frequencies)}Hz would need at least {n_cycles*1/np.min(frequencies)} seconds of data")

    complexData = np.zeros((n_freqs, nTrials, nChans, output_len), dtype=complex)

    for f_idx, frequency in enumerate(frequencies):
        wavelet_len = int(n_cycles * 1000.0 / (frequency * dt_ms))
        pad_cur = pad_min - int(n_cycles * 500.0 / (frequency * dt_ms))

        window = tapered_gaussian(wavelet_len)
        phase_wavelet = np.conj(np.exp(1j * (2.0 * np.pi * n_cycles * np.arange(wavelet_len) / wavelet_len))) * window

        for trl in range(nTrials):
            for chan in range(nChans):
                segment_view = sliding_window_view(data[trl, chan, :], wavelet_len)
                segments = segment_view[pad_cur:pad_cur + output_len] 
                segMean = segments.mean(axis=1, keepdims=True)
                convolved = np.sum((segments-segMean) * phase_wavelet[np.newaxis, :], axis=1)
                complexData[f_idx, trl, chan, :] = convolved
    
    if hasBeenReshaped:
        index = currentDimord.split("_").index("time")
        temp = list(oldshape)
        temp[index] = output_len  
        oldshape = tuple(temp)
        complexData = np.reshape(complexData, (len(frequencies), *oldshape))
    currentDimord = "freq_" + currentDimord    
    time = waveData.get_time(dataBucketName)[pad_min:-pad_min]
    print(f"Warning: this function uses {pad_min} samples of padding, output dataBucket will be shorter by twice that")
    complexDataBucket = wd.DataBucket(complexData, "complexData", currentDimord,time=time,chanNames=waveData.get_channel_names())
    waveData.add_data_bucket(complexDataBucket)

def gaussian(x, a, x0, sigma):
    """
    Parameters
    ----------
    x : numpy.ndarray
    a : float
    x0 : float
    sigma : float

    Returns
    -------
    numpy.ndarray
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def tapered_gaussian(n):
    """
    Parameters
    ----------
    n : int

    Returns
    -------
    numpy.ndarray
    """
    x = np.arange(n)
    taper = 0.5 * (1.0 - np.cos(2 * np.pi * x / (n - 1)))
    mean = np.sum(x * taper) / n
    sigma = np.sqrt(np.sum(taper * (x - mean) ** 2) / n)
    popt, _ = curve_fit(gaussian, x, taper, p0=[1, mean, sigma])
    return gaussian(x, *popt)