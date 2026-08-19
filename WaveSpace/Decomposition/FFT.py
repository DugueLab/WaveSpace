#%%
import numpy as np
from scipy.fftpack import fft, fftfreq
from scipy.signal.windows import hann

from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wd


#%%
def hann_fft(waveData, dataBucketName = "", timeStart = None, timeEnd =  None, timeStep = 1, freqStart = 0, freqEnd = -1):
    """Compute a Hann-windowed fast Fourier transform over time.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to transform.
    dataBucketName : str, default=""
        Name of the input data bucket. Defaults to the active data bucket.
    timeStart : float or list, default=[]
        Start time in seconds. Defaults to the first sample.
    timeEnd : float or list, default=[]
        End time in seconds. Defaults to the last sample.
    timeStep : int, default=1
        Sample stride applied before the transform.
    freqStart : float, default=0
        Lowest retained frequency in Hz.
    freqEnd : float, default=-1
        Highest retained frequency in Hz. Defaults to the Nyquist frequency (half the sample rate).
    Returns
    -------
    None
        Adds complex Fourier coefficients to ``waveData`` as the ``FFT`` data
        bucket.

    Notes
    -----
    The transform is applied along the ``time`` dimension after data are
    temporarily reordered to trial, channel, and time dimensions.
    """
    #set defaults
    if timeEnd is None:
        timeEnd = []
    if timeStart is None:
        timeStart = []
    if timeStart == []:
        timeStart = waveData.get_time()[0]
    if timeEnd == []:
        timeEnd = waveData.get_time()[-1]

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
    
    # Select time range (input is in seconds)
    timeStart = int(timeStart * waveData.get_sample_rate())
    timeEnd = int(timeEnd * waveData.get_sample_rate())
    
    currentData = currentData[..., timeStart:timeEnd:timeStep]
    
    # Compute FFT
    nSamples = currentData.shape[-1]
    hannWindow = hann(nSamples)
    #freq resolution
    freqStep = waveData.get_sample_rate() / nSamples
    #freq range
    if freqEnd == -1:
        freqEnd = waveData.get_sample_rate() / 2
    freqStart = int(freqStart / freqStep)
    freqEnd = int(freqEnd / freqStep)
    freqs = fftfreq(nSamples, 1/waveData.get_sample_rate())[freqStart:freqEnd]
    fft_result = fft(currentData * hannWindow, axis=-1)[:, :, freqStart:freqEnd]
    #normalize
    fft_result = fft_result / nSamples
      

    
    if hasBeenReshaped:
        fft_result = np.reshape(fft_result, (freqs.shape(),*origShape))

    complexDataBucket = wd.DataBucket(fft_result, "FFT", origDimord.replace('time', 'freq'),
                                       time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time(),
                                       chanNames= waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(complexDataBucket)
    waveData.log_history(["Frequency Decomposition", "FFT", "Time range: " + str(timeStart) + " to " + str(timeEnd) + " in steps of " + str(timeStep), "Frequency range: " + str(freqStart) + " to " + str(freqEnd) + " in steps of " + str(freqStep)])
