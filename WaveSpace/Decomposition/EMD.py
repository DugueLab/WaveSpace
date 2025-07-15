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
    emd.logger.set_up(level='CRITICAL')  # supress the warning about too few IMFs

# Utility funs________________________________________________
def FreqAmpPhaseFromAnalytic(waveData, smooth_phase=None, smooth_freq = 3, dataBucketName="", timeRange=(slice(None))):
    """Get the instantaneous frequency, amplitude, and phase from the analytic signal.
    WaveData: waveData object
    smooth_phase: smoothing window for phase
    smooth_freq: smoothing window for frequency
    dataBucketName: name of the data bucket to use. Defaults to the active data bucket
    timeRange: time range to use in samples. Defaults to all time points
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
    """Plot the frequency spectrum of the IMFs.
    IA: instantaneous amplitude
    IF: instantaneous frequency
    waveData: waveData object
    freqMin: minimum frequency to plot
    freqMax: maximum frequency to plot
    nbins: number of bins for the histogram, defaults to 50
    trialnum: trial number to plot, defaults to 0
    chanNum: channel number to plot, defaults to 0
    FOI: frequency of interest, draws a vertical line at FOI, defaults to None
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
    IPs, IFs, IAs, base_imf = args
    return assess_harmonic_criteria(IPs.T, IFs.T, IAs.T, base_imf=base_imf, num_segments=10)

def check_for_harmonics(waveData, IPs, IFs, IAs, base_imf_list):
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
        # Create a pool of workers
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            # Distribute the work among the workers
            HarmonicInds = pool.map(parallel_assess_harmonic_criteria, args)
            HarmonicInds = [[HarmonicInds[i * num_channels + j] for j in range(num_channels)] for i in range(num_trials)]
    else:
        # Use joblib for parallelization if not on Linux
        HarmonicInds = joblib.Parallel(n_jobs=joblib.cpu_count())(joblib.delayed(parallel_assess_harmonic_criteria)(arg) for arg in args)
        HarmonicInds = [[HarmonicInds[i * num_channels + j] for j in range(num_channels)] for i in range(num_trials)]
    
    return HarmonicInds

def assess_harmonic_criteria(IP, IF, IA, num_segments=None, base_imf=None, print_result=True):
    """Assess IMFs for potential harmonic relationships.

    This function implements tests for the criteria defining when signals can
    be considered 'harmonically related' as introduced in [1]_. Broadly,
    harmonically related signals are defined as having an integer frequency
    ratio, constant phase relationship, and a well-defined joint instantaneous
    frequency

    Three criteria are assessed by splitting the time-series into approximately
    equally sized segments and computing metrics within each segment.

    Parameters
    ----------
    IP, IF, IA : ndarray of equal shape
        Instantaneous Phase, Frequency and Amplitude estimates for a set of
        IMFs. These are typically the outputs from emd.spectra.frequency_transform.
    num_segments : int
        Number of segments to split the time series into to enable statistical assessment.
    base_inf : int
        Index of IMF to be considered the potential 'fundamental' oscillation.
    print_result : bool
        Flag indicating whether to print a summary table of results.

    Returns
    -------
    df
        Pandas DataFrame containing a range of summary and comparison metrics.

    Notes
    -----
    In detail, this function compares each IMF to a 'base' IMF to see if it can
    be considered a potential harmonic. Each pair of IMFs are assessed for:

    1) An integer frequency ratio. The distribution of frequency ratios across
    segments is compared to its closest integer value with a 1-sample t-test

    2) Consistent phase relationship. The instantaneous phase time-courses are
    assessed for temporal dependence using a Distance Correlation t-statistic.

    3) The af ratio is less than 1. The product of the amplitude ratio and
    frequency ratio of the two IMFs should be less than 1 according to a
    1-sided 1-sample t-test.

    References
    ----------
    .. [1] Fabus, M. S., Woolrich, M. W., Warnaby, C. W., & Quinn, A. J.
           (2022). Understanding Harmonic Structures Through Instantaneous Frequency.
           IEEE Open Journal of Signal Processing. doi: 10.1109/OJSP.2022.3198012.

    """
    # Housekeeping
    import dcor
    import pandas as pd
    from scipy.stats import ttest_1samp
    IP, IF, IA = emd.imftools.ensure_2d([IP, IF, IA], ['IP', 'IF', 'IA'], 'assess_harmonic_criteria')
    emd.imftools.ensure_equal_dims((IP, IF, IA), ('IP', 'IF', 'IA'), 'assess_harmonic_criteria')

    if base_imf is None:
        base_imf = IP.shape[1] - 1

    IP = IP.copy()[:, :base_imf+1]
    IF = IF.copy()[:, :base_imf+1]
    IA = IA.copy()[:, :base_imf+1]

    if num_segments is None:
        num_segments = 20
    mod = int(len(IP) % num_segments) #KP: check if dividing into num_segments produces a rest, if so, remove those leftover samples from array
    # the [:-mod or None] expression means return everything except the rest of the division (mod) if mod is zero, return everything
    IPs = np.array_split(IP[:-mod or None], num_segments, axis=0)
    IFs = np.array_split(IF[:-mod or None], num_segments, axis=0)
    IAs = np.array_split(IA[:-mod or None], num_segments, axis=0)

    IFms = [ff.mean(axis=0) for ff in IFs]
    IAms = [aa.mean(axis=0) for aa in IAs]

    fratios = np.zeros((base_imf, num_segments))
    a_s = np.zeros((base_imf, num_segments))
    afs = np.zeros((base_imf, num_segments))
    dcorrs = np.zeros((base_imf, num_segments))
    dcor_pvals = np.zeros((base_imf, 2))
    fratio_pvals = np.zeros(base_imf)
    af_pvals = np.zeros(base_imf)

    for ii in range(base_imf):
        # Freq ratios
        fratios[ii, :] = [ff[ii] / ff[base_imf] for ff in IFms]
        # Amp ratio
        a_s[ii, :] = [aa[ii] / aa[base_imf] for aa in IAms]
        # af value
        afs[ii, :] = a_s[ii, :] * fratios[ii, :]

        # Test 1: significant Phase-Phase Correlation
        dcorr = dcor.distance_correlation(IP[:, ii], IP[:, base_imf])
        p_dcor, _ = dcor.independence.distance_correlation_t_test(IP[:, ii], IP[:, base_imf])
        dcor_pvals[ii, :] = dcorr, p_dcor
        for jj in range(num_segments):
            dcorrs[ii, jj] = dcor.distance_correlation(IPs[jj][:, ii], IPs[jj][:, base_imf])

        # Test 2: frequency ratio not different from nearest integer
        ftarget = np.round(fratios[ii, :].mean())
        _, fratio_pvals[ii] = ttest_1samp(fratios[ii, :], ftarget)
        # Test 3: af < 1
        _, af_pvals[ii] = ttest_1samp(afs[ii, :], 1, alternative='less')

    info = {'InstFreq Mean': np.array(IFms).mean(axis=0)[:base_imf],
            'InstFreq StDev': np.array(IFms).std(axis=0)[:base_imf],
            'InstFreq Ratio': fratios.mean(axis=1),
            'Integer IF p-value': fratio_pvals,
            'InstAmp Mean': np.array(IAms).mean(axis=0)[:base_imf],
            'InstAmp StDev': np.array(IAms).std(axis=0)[:base_imf],
            'InstAmp Ratio': a_s.mean(axis=1),
            'af Value': afs.mean(axis=1),
            'af < 1 p-value': af_pvals,
            'DistCorr': dcor_pvals[:, 0],
            'DistCorr p-value': dcor_pvals[:, 1]}

    df = pd.DataFrame.from_dict(info)

    # KP added: set some criteria for when to merge IMFs, return indices of IMFs that meet criteria
    # loop over the IMFs that are not the base
    condition = (df['Integer IF p-value'] < .01) \
        & (df['af < 1 p-value'] < .01) \
        & (df['DistCorr p-value'] < .01)
    HarmonicInds = df.index.values[condition]
    return HarmonicInds

def CombineIMFsIfPositiveJointInstFreq(waveData, potentialHarmonicInds, dataBucketName=""):
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
    """Find the index and value of the element in IF closest to FOI
    Args:
        waveData (WaveData object)
        IF (np.ndarray)
        FOI (float)
        start_time (float)
        end_time (float)
    Returns:
        ind (np.ndarray)
        value (np.ndarray)
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
    """Empirical mode decomposition. Wrapper function for emd.sift.** from emd package.
    Note that. to speed things up a little, this function uses multiprocessing with numpy arrays. The number of intrinsic
    mode functions that are actually found in the data may be less than the number of IMFs requested for any given timeseries.
    Those rows of the output array will be filled with NaNs. If you have a better idea for how to do this, please let me know.

    Args:
        waveData (waveData object)
        nIMFs (int): max number of IMFs to extract. Defaults to 7.
        type (str, optional): Defaults to "regular".Options are masked_sift, iterated_masked_sift, ensemble_sift, multivariate_sift
            for more info, check out the excellent documentation at https://emd.readthedocs.io/en/stable/index.html
        n_noiseChans (int, optional): Defaults to 10. Number of noise channels to add for multivariate siftfun
        noiseVar (float, optional): Defaults to 0.05. Variance of noise to add for multivariate siftfun
        dataBucketName (str, optional):Defaults to ""
        ndir (int, optional): Defaults to None. Number of signal projections. Should be at least twice the number of data channels. Only for multivariate siftfun
        sd (float, optional): Defaults to 0.075. Only for multivariate siftfun
        sd2 (float, optional): Defaults to 0.75. Only for multivariate siftfun
        tol (float, optional): Defaults to 0.075. Only for multivariate siftfun
        stp_crit (str, optional): Defaults to 'stop'. Only for multivariate siftfun. Options are 'stop', 'fix_h'
        stp_cnt (int, optional): Defaults to 2. Only for multivariate siftfun and only of stp_crit is 'fix_h'


    Returns: changes the waveData object in place. Adds a new data bucket called "AnalysticSignal" 
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
    trl, chn = pair
    imf = siftfun(
        currentData[trl, chn, :], max_imfs=nIMFs, verbose="CRITICAL")
    analytic_signal = signal.hilbert(imf, axis=0)
    #maxAmpind = np.nanargmax(np.median(analytic_signal, axis=0))    
    return analytic_signal.T