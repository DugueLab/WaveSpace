from WaveSpace.Utils import WaveData as wd
from WaveSpace.Utils import HelperFuns as hf

import emd
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import multiprocessing
from itertools import product
import joblib
import platform


def start_emd_logging():
    """Returns
    -------
    None
    """
    emd.logger.set_up(level='CRITICAL')  # supress the warning about too few IMFs

# Utility funs________________________________________________
def FreqAmpPhaseFromAnalytic(waveData, smooth_phase=None, smooth_freq = 3, dataBucketName="", timeRange=(slice(None))):
    """Parameters
    ----------
    waveData : WaveData
    smooth_phase : int or None, default=None
    smooth_freq : int, default=3
    dataBucketName : str, default=""
    timeRange : tuple, default=(slice(None),)

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
    """

    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    oldShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "imf_trl_chan_time")
    # Get the number of trials, channels, and IMFs
    if not timeRange == (slice(None)):
        currentData = currentData[:,:,:,timeRange[0]:timeRange[1]]
    nIMFs, nTrials, nChans,  nTime = currentData.shape
    IA = np.full((nIMFs,nTrials, nChans,  nTime), np.nan, dtype=float)
    IF = np.full((nIMFs,nTrials, nChans, nTime), np.nan, dtype=float)
    IP = np.full((nIMFs,nTrials, nChans,  nTime), np.nan, dtype=float)
    # Loop over trials, channels, and IMFs
    for trl in range(nTrials):
        for chan in range(nChans):
            iphase = emd.spectra.phase_from_complex_signal(currentData[:, trl, chan].T,
                                                           smoothing=smooth_phase, phase_jump='peak', ret_phase='unwrapped')
            if np.isnan(iphase[0,:]).any():
                ind = np.where(np.isnan(iphase[0,:]))
                tempIF = emd.spectra.freq_from_phase(
                    iphase[:, 0:np.min(ind)], waveData.get_sample_rate(), savgol_width=smooth_freq).T
                nIMFs_new = tempIF.shape[0]
                IF[:nIMFs_new, trl,chan, :] = tempIF
                IP[:nIMFs_new, trl, chan, :] = emd.imftools.wrap_phase(iphase[:, 0:np.min(ind)]).T
                IA[:, trl, chan] = np.abs(currentData[:, trl, chan])
            else: 
                IF[:,trl, chan] = emd.spectra.freq_from_phase(
                iphase, waveData.get_sample_rate(), savgol_width=smooth_freq).T
                IP[:,trl, chan] = emd.imftools.wrap_phase(iphase).T
                IA[:,trl, chan] = np.abs(currentData[:,trl, chan])
    if hasBeenReshaped:        
        IF = np.reshape(IF, (oldShape))
        IA = np.reshape(IA, (oldShape))
        IP = np.reshape(IP, (oldShape))
    return IF, IA, IP

def checkFrequencySpectrum(IA, IF, waveData, freqMin, freqMax, nbins=50, trialnum=0, chanNum=0, FOI = None):
    """Parameters
    ----------
    IA : numpy.ndarray
    IF : numpy.ndarray
    waveData : WaveData
    freqMin : float
    freqMax : float
    nbins : int, default=50
    trialnum : int, default=0
    chanNum : int, default=0
    FOI : float or None, default=None

    Returns
    -------
    tuple[matplotlib.figure.Figure, numpy.ndarray]
    """
    currentDimord = waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    hf.assure_consistency(waveData)
    IAhasBeenReshaped, IA =  hf.force_dimord(IA, currentDimord , "imf_trl_chan_time")
    IFhasBeenReshaped, IF =  hf.force_dimord(IF, currentDimord , "imf_trl_chan_time")
    IA = IA[:,trialnum, chanNum].T
    IF = IF[:,trialnum, chanNum].T
    #remove potential nan IMFs
    IA = IA[:, ~np.all(np.isnan(IA), axis=0)]
    IF = IF[:, ~np.all(np.isnan(IF), axis=0)]
    freqEdges, freqCenters = emd.spectra.define_hist_bins(
        freqMin, freqMax, nbins)
    freqRes = freqCenters[1]-freqCenters[0]
    f, spec_weighted = emd.spectra.hilberthuang(
        IF, IA, freqEdges, sample_rate=waveData.get_sample_rate(), sum_imfs=False)
    f, spec_unweighted = emd.spectra.hilberthuang(IF, np.ones_like(
        IA), freqEdges, sample_rate=waveData.get_sample_rate(), sum_imfs=False)
    fig = plt.figure(figsize=(10, 4))
    plt.subplots_adjust(hspace=0.4)
    plt.subplot(121)
    plt.plot(freqCenters, spec_unweighted)
    plt.vlines(FOI, 0, np.max(spec_unweighted), color='k', linestyle='--')
    plt.xticks(np.arange(10)*10)
    if FOI:
        plt.vlines(FOI, 0, np.max(spec_unweighted), color='k', linestyle='--')
        label_y = np.max(spec_unweighted) - 0.05 * np.max(spec_unweighted)  # Slightly below the max
        label_x_offset = 1  # Adjust this as per your needs
        plt.annotate(str(FOI) + 'Hz', xy=(FOI, label_y), xytext=(FOI + label_x_offset, label_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                verticalalignment='center')
    plt.xlim(0, freqMax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('unweighted\nHilbert-Huang Transform')

    plt.subplot(122)
    plt.plot(freqCenters, spec_weighted)
    plt.xticks(np.arange(10)*10)
    if FOI:
        plt.vlines(FOI, 0, np.max(spec_weighted), color='k', linestyle='--')
        label_y = np.max(spec_weighted) - 0.05 * np.max(spec_weighted)  # Slightly below the max
        label_x_offset = 1  # Adjust this as per your needs
        plt.annotate(str(FOI) + 'Hz', xy=(FOI, label_y), xytext=(FOI + label_x_offset, label_y),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                verticalalignment='center')
    plt.xlim(0, freqMax)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('IA-weighted\nHilbert-Huang Transform')
    plt.show()
    return fig, spec_weighted
   
def freqSpecTrialAverage(waveData, freqMin, freqMax, nbins=50, dataBucketName = "", timeRange =(slice(None))):
    """Parameters
    ----------
    waveData : WaveData
    freqMin : float
    freqMax : float
    nbins : int, default=50
    dataBucketName : str, default=""
    timeRange : tuple, default=(slice(None),)

    Returns
    -------
    list[numpy.ndarray]
    """
    freqEdges, freqCenters = emd.spectra.define_hist_bins(
        freqMin, freqMax, nbins)
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    assert (len(waveData.DataBuckets[dataBucketName].get_dimord().split("_"))==4) , "Inputdata must have 4 dimensions of order imf_trl_chan_time"
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    imf, trials, channels, time = currentData.shape
    sample_rate=waveData.get_sample_rate()
    AllSpecWeighted = np.zeros((channels, nbins))
    tempSpec = np.zeros((trials, nbins))    
    IF, IA, IP = FreqAmpPhaseFromAnalytic(waveData, 5, 3, dataBucketName="", timeRange = timeRange)
    
    #prepare arguments for parallel
    args = [(IF[:, :, channel, : ], IA[:, :, channel, : ], trials, freqEdges, sample_rate, tempSpec) for channel in range(channels)]
    if platform.system() == 'Linux':
        # Create a pool of workers
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            # Distribute the work among the workers
            result = pool.map(freqSpecTrialAverageProcessChannel, args)
    else:
        # Use joblib for parallelization if not on Linux
        result = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(freqSpecTrialAverageProcessChannel)(arg) for arg in args)
    return result

def freqSpecTrialAverageProcessChannel(args):
    """Parameters
    ----------
    args : tuple

    Returns
    -------
    numpy.ndarray
    """
    IF, IA, trials, freqEdges, sample_rate, tempSpec = args
    for trl in range(trials):
        if np.isnan(IF[:,trl,0]).any():
            ind = np.min(np.where(np.isnan(IF)))
        else:
            ind= IF.shape[0]+1
        f, IMFspectrum = emd.spectra.hilberthuang(
                IF[0:ind,trl,:].T, IA[0:ind,trl, :].T, freqEdges, sample_rate= sample_rate , sum_imfs=True)
        tempSpec[trl,:] = IMFspectrum
    AllSpecWeighted = np.mean(tempSpec, axis=0)    
    return AllSpecWeighted

def parallel_assess_harmonic_criteria(args):
    """Parameters
    ----------
    args : tuple

    Returns
    -------
    pandas.DataFrame
    """
    IPs, IFs, IAs, base_imf = args
    return emd.imftools.assess_harmonic_criteria(IPs.T, IFs.T, IAs.T, base_imf=base_imf, num_segments=10)

def check_for_harmonics(waveData, IPs, IFs, IAs, base_imf_list):
    """Parameters
    ----------
    waveData : WaveData
    IPs : numpy.ndarray
    IFs : numpy.ndarray
    IAs : numpy.ndarray
    base_imf_list : numpy.ndarray

    Returns
    -------
    list[list[numpy.ndarray]]
    """
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    IAhasBeenReshaped, IAs =  hf.force_dimord(IAs, currentDimord , "imf_trl_chan_time")
    IFhasBeenReshaped, IFs =  hf.force_dimord(IFs, currentDimord , "imf_trl_chan_time")
    pairs = list(product(range(IPs.shape[1]), range(IPs.shape[2])))
    num_trials, num_channels = IPs.shape[1:3]
    args = [list((IPs[:, pair[0], pair[1]], IFs[:, pair[0], pair[1]], IAs[:, pair[0], pair[1]], base_imf_list[pair[0], pair[1]])) for pair in pairs]
    #make sure there are no nan IMFs    
    for argnum, arg in enumerate(args):
        if np.isnan(arg[0]).any():
            ind = np.logical_not( np.isnan(arg[0]).all(axis=1))
            args[argnum][0] = arg[0][ind][:]
            args[argnum][1] = arg[1][ind][:]
            args[argnum][2] = arg[2][ind][:] 

    if platform.system() == 'Linux':
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            df = pool.map(parallel_assess_harmonic_criteria, args)    
            # loop over the IMFs that are not the base
            condition = (df['Integer IF p-value'] < .01) \
                & (df['af < 1 p-value'] < .01) \
                & (df['DistCorr p-value'] < .01)
            HarmonicInds = df.index.values[condition]
            HarmonicInds = [[HarmonicInds[i * num_channels + j] for j in range(num_channels)] for i in range(num_trials)]
    else:
        # Use joblib for parallelization if not on Linux
        df = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(parallel_assess_harmonic_criteria)(arg) for arg in args)
        # KP added: set some criteria for when to merge IMFs, return indices of IMFs that meet criteria
        # loop over the IMFs that are not the base
        condition = (df['Integer IF p-value'] < .01) \
            & (df['af < 1 p-value'] < .01) \
            & (df['DistCorr p-value'] < .01)
        HarmonicInds = df.index.values[condition]
        HarmonicInds = [[HarmonicInds[i * num_channels + j] for j in range(num_channels)] for i in range(num_trials)]
    return HarmonicInds   

def CombineIMFsIfPositiveJointInstFreq(waveData, potentialHarmonicInds, dataBucketName=""):
    """Parameters
    ----------
    waveData : WaveData
    potentialHarmonicInds : list[list[numpy.ndarray]]
    dataBucketName : str, default=""

    Returns
    -------
    None
    """
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    currentData = waveData.get_data(dataBucketName)
    origShape = currentData.shape
    hasbeenreshaped, currentData = hf.force_dimord(currentData, waveData.DataBuckets[dataBucketName].get_dimord(), "imf_trl_chan_time")
    for trl in range(currentData.shape[1]):
        for chan in range(currentData.shape[2]):
            # check if HarmonicInds has more than one value, if so, combine them
            if len(potentialHarmonicInds[trl][chan]) > 1 :
                jif = currentData[potentialHarmonicInds[trl][chan][0],trl, chan, :]\
                    + currentData[potentialHarmonicInds[trl][chan][1],trl, chan,  :]
                _, IF, _ = emd.spectra.frequency_transform(np.real(jif).T, 1, 'hilbert')
                if not any(IF < 0):
                    print('Hamonic IMF found, combining IMFs')
                    currentData[potentialHarmonicInds[trl][chan][0], trl, chan, :] = signal.hilbert(np.real(jif))
                    currentData[potentialHarmonicInds[trl][chan][1], trl, chan, :] = np.nan

    if hasbeenreshaped:
        currentData = np.reshape(currentData, origShape)
    complexDataBucket = wd.DataBucket(currentData, "complexData", waveData.DataBuckets[dataBucketName].get_dimord(),time=waveData.DataBuckets[dataBucketName].get_time() , chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(complexDataBucket)
    waveData.log_history(["EMD", "Combined harmonic IMFs"])

def find_nearest_to_FOI(waveData, IF, FOI, start_time=None, end_time=None):
    """Parameters
    ----------
    waveData : WaveData
    IF : numpy.ndarray
    FOI : float
    start_time : float or None, default=None
    end_time : float or None, default=None

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
    """
    hf.assure_consistency(waveData)
    currentDimord = waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    hasBeenReshaped, IF =  hf.force_dimord(IF, currentDimord , "imf_trl_chan_time")
    shape = IF.shape
    ind = np.zeros((shape[1], shape[2]),dtype=int)
    meanFreq = np.zeros((shape[1], shape[2]))
    time = waveData.get_time()

    # If no specific time range is provided, use all time points
    if start_time is None:
        start_time = time[0]
    if end_time is None:
        end_time = time[-1]

    # Find indices corresponding to start and end times
    start_ind, _ = hf.find_nearest(time, start_time)
    end_ind, _  = hf.find_nearest(time, end_time)

    for trl in range(shape[1]):
        for chan in range(shape[2]):
            # Use only time points within specified range
            segment = IF[:, trl, chan, start_ind:end_ind+1]
            mean_freqs = hf.trim_mean(segment, .1, axis=-1)
            ind[trl, chan], meanFreq[trl, chan] = hf.find_nearest(mean_freqs, FOI)

    return ind, meanFreq

    #________________________________________________
    #KP test, not finished!!!

#main EMD funs
def EMD(waveData, nIMFs=7, dataBucketName="", noiseVar = 0.05, n_noiseChans = 10, siftType = 'regular',
        ndir=None, stp_crit ='stop', sd=0.075, sd2=0.75, tol=0.075,stp_cnt=2):
    """Parameters
    ----------
    waveData : WaveData
    nIMFs : int, default=7
    dataBucketName : str, default=""
    noiseVar : float, default=0.05
    n_noiseChans : int, default=10
    siftType : str, default="regular"
    ndir : int or None, default=None
    stp_crit : str, default="stop"
    sd : float, default=0.075
    sd2 : float, default=0.75
    tol : float, default=0.075
    stp_cnt : int, default=2

    Returns
    -------
    None
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
    #set EMD fun to use based on siftType
    if siftType == "masked_sift":
        siftfun = emd.sift.mask_sift
    elif siftType == "iterated_masked_sift":
        siftfun = emd.sift.iterated_mask_sift
    elif siftType == "ensemble_sift":
        siftfun = emd.sift.ensemble_sift
    elif siftType == "multivariate_sift":
        from WaveSpace.Decomposition import MEMD_Matlab_translation as MEMD
        #uses de Souza e Silva translation from Matlab to Python
        #original is based on: Rehman and D. P. Mandic, "Multivariate Empirical Mode Decomposition", Proceedings of the Royal Society A, 2010
        #KP added some noise channels, see: ur Rehman, N., Park, C., Huang, N. E., & Mandic, D. P. (2013). EMD via MEMD: multivariate noise-aided computation of standard EMD. Advances in adaptive data analysis, 5(02), 1350007.
        siftfun = MEMD.memd 
    else:
        siftfun = emd.sift.sift
    print("Using siftfun: ", siftfun)
    nTimeseries, nChans, nTime = currentData.shape
    complexData = np.full((nIMFs,nTimeseries, nChans, nTime),
                            np.nan, dtype=np.complex64)
    
    if siftType=='multivariate_sift':
        #add noise channels. Those are just for the multivariate siftfun and will be removed before returning the data
        currentData = hf.add_noise_channels(currentData, proportion=noiseVar, No_noise_channels=n_noiseChans)
      
        for trl in range(nTimeseries):
            print('now processing trial: ' + str(trl))
            if ndir is None:
                ndir = 2*nChans
            result = MEMD.memd(x=currentData[trl].T, ndir=ndir, maxnIMF=nIMFs, stp_crit =stp_crit, sd=sd, sd2=sd2, tol=tol,stp_cnt = stp_cnt)
            result = result[:nIMFs,:nChans,:] #cut off any imfs exeeding maxnIMFs and remove the noise channels that were added above
            complexData[:result.shape[0],trl, :, :] = signal.hilbert(result)

    else:

        if platform.system() == 'Linux':
            # split by channels
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()-1) as pool:
                pairs = [(trl, chn) for trl in range(nTimeseries) for chn in range(nChans)]
                results = pool.starmap(EMD_process_timeseries, [(pair, currentData, nIMFs, siftfun) for pair in pairs])

        else:  # Assuming other OS is Windows here
            # split by channels
            with joblib.Parallel(n_jobs=joblib.cpu_count()-1) as parallel:
                pairs = [(trl, chn) for trl in range(nTimeseries) for chn in range(nChans)]
                results = parallel(joblib.delayed(EMD_process_timeseries)(pair, currentData, nIMFs, siftfun) for pair in pairs)

        for pair, result in zip(pairs, results):
            trl, chn = pair
            complexData[:result.shape[0],trl, chn, :] = result[:nIMFs]

    if hasBeenReshaped:
        complexData = np.reshape(complexData, (nIMFs,*origShape))

    complexDataBucket = wd.DataBucket(complexData, "complexData", "IMF_" + origDimord,
                                      time=waveData.DataBuckets[waveData.ActiveDataBucket].get_time()
                                      ,chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(complexDataBucket)
    waveData.log_history(["Phase estimate", "EMD","siftType: " , siftType, "nIMFS: ", nIMFs])

def EMD_process_timeseries(pair, currentData, nIMFs, siftfun):
    """Parameters
    ----------
    pair : tuple[int, int]
    currentData : numpy.ndarray
    nIMFs : int
    siftfun : callable

    Returns
    -------
    numpy.ndarray
    """
    trl, chn = pair
    imf = siftfun(
        currentData[trl, chn, :], max_imfs=nIMFs, verbose="CRITICAL")
    analytic_signal = signal.hilbert(imf, axis=0)
    #maxAmpind = np.nanargmax(np.median(analytic_signal, axis=0))    
    return analytic_signal.T