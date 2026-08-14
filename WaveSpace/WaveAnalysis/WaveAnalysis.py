import copy
import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from WaveSpace.Utils import HelperFuns
from WaveSpace.Utils import WaveData as wd


def FFT_2D(waveData, channelIndices, lowerBound, upperBound, DataBucketName = ""):
    #code is adapted from publicly available Matlab code written by Andrea Alamia (available here: https://github.com/artipago/travellingWaveEEG)
    #corresponding publication: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3000487
    # 'Along' is the max value in the upper right quadrant of the 2D FFT trimmed to lower and upper bound
    # 'Reverse' is the max value in the lower right quadrant of the 2D FFT trimmed to lower and upper bound
    # those correspond to a phase-shift along or opposite to the direction of selected channels, respectively
    # i.e. along means wave travels along direction of selected channels
    if DataBucketName != "":
        waveData.set_active_dataBucket(DataBucketName)

    HelperFuns.assure_consistency(waveData)
    shape = waveData.get_active_data().shape
    if isinstance(channelIndices[0], (list, tuple)):
        if len(shape) == 3:
            raise Exception("two dimensional index used for one dimensional channel data")
        dataIDX = (slice(None), [i[0] for i in channelIndices], [i[1] for i in channelIndices], slice(None))
        filteredData = waveData.get_active_data()[dataIDX]   
        waveData.get_channel_positions()

    else:
        if len(shape) == 4:
            raise Exception("one dimensional index used for two dimensional channel data")
        filteredData = waveData.get_active_data()[:,channelIndices,:]  
    
    # Define the spatial distances and temporal sampling rate
    dist_bw_channels = 1  # distance between channels
    sampling_rate = waveData.get_sample_rate()

    nTrials, nChan, nTimepoints = filteredData.shape 

    # Frequency axes
    tempFreqAxis = np.fft.fftfreq(nTimepoints, 1.0/sampling_rate)
    tempFreqAxis = np.fft.fftshift(tempFreqAxis)  # Shift to range -sfreq/2 to sfreq/2

    spatialFreqAxis = np.fft.fftfreq(nChan, dist_bw_channels)
    spatialFreqAxis = np.fft.fftshift(spatialFreqAxis)  # Shift to center

    TemporalRange = np.where((tempFreqAxis >= lowerBound) & (tempFreqAxis <= upperBound))[0]

    # Arrays to store results
    allMaxAlongvalue = np.zeros(nTrials)
    allMaxReversevalue = np.zeros(nTrials)
    allAlongTempFreq = np.zeros(nTrials)
    allReverseTempFreq = np.zeros(nTrials)
    allAlongSpatFreq = np.zeros(nTrials)
    allReverseSpatFreq = np.zeros(nTrials)
    allFFT_abs = np.zeros((nTrials, nChan, nTimepoints))

    for trialNr in range(nTrials):
        data = filteredData[trialNr,:,:]
        FFT_abs = np.abs(np.fft.fftshift(np.fft.fft2(data)))**2
        allFFT_abs[trialNr,:,:] = FFT_abs

        # Find max for spatialFreq > 0
        SpatialRange = np.where(spatialFreqAxis > 0)[0]
        TrimmedFFT = FFT_abs[SpatialRange,:][:,TemporalRange]
        maxReversevalue = TrimmedFFT.max()
        maxReverse_inds = np.unravel_index(TrimmedFFT.argmax(), TrimmedFFT.shape)
        bwTempFreq = tempFreqAxis[TemporalRange[maxReverse_inds[1]]]
        bwSpatFreq = spatialFreqAxis[SpatialRange[maxReverse_inds[0]]]
        
        # Find max for spatialFreq < 0
        SpatialRange = np.where(spatialFreqAxis < 0)[0]
        TrimmedFFT = FFT_abs[SpatialRange,:][:,TemporalRange]
        maxAlongvalue = TrimmedFFT.max()
        maxAlong_inds = np.unravel_index(TrimmedFFT.argmax(), TrimmedFFT.shape)
        AlongTempFreq = tempFreqAxis[TemporalRange[maxAlong_inds[1]]]
        AlongSpatFreq = spatialFreqAxis[SpatialRange[maxAlong_inds[0]]]
        
        # Store results
        allMaxAlongvalue[trialNr] = maxAlongvalue
        allMaxReversevalue[trialNr] = maxReversevalue
        allAlongTempFreq[trialNr] = AlongTempFreq 
        allReverseTempFreq[trialNr] = bwTempFreq
        allAlongSpatFreq[trialNr] =  AlongSpatFreq
        allReverseSpatFreq[trialNr] = bwSpatFreq
    	
    info = {
        'Max Along Power': allMaxAlongvalue,
        'Max Reverse Power': allMaxReversevalue,
        'Temporal Frequency at Max Along': allAlongTempFreq,
        'Temporal Frequency at Max Reverse': allReverseTempFreq,
        'Spatial Frequency at Max Along': allAlongSpatFreq,
        'Spatial Frequency at Max Reverse': allReverseSpatFreq}

    df = pd.DataFrame.from_dict(info)  
    fftBucket = wd.DataBucket(allFFT_abs,"FFT_ABS","trl_spatfreq_tempfreq", chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(fftBucket)
    resultBucket = wd.DataBucket(df, "Result", "2D_FFT", chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(resultBucket)


def FFT_2D_shuffle_and_average(waveData, channelIndices, lowerBound, upperBound,num_iterations = 100, DataBucketName = ""):
    if DataBucketName != "":
        waveData.set_active_dataBucket(DataBucketName)
    HelperFuns.assure_consistency(waveData)
    shape = waveData.get_active_data().shape
    if isinstance(channelIndices[0], (list, tuple)):
        if len(shape) == 3: 
            raise Exception("two dimensional index used for one dimensional channel data")
        (slice(None), [i[0] for i in channelIndices], [i[1] for i in channelIndices], slice(None))
        waveData.get_channel_positions()
        channelNames = [waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names()[idx[0],idx[1]] for idx in channelIndices]

    else:
        if len(shape) == 4:
            raise Exception("one dimensional index used for two dimensional channel data")
        channelNames = [waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names()[idx] for idx in channelIndices]

    # Create a deep copy of the waveData object to not mess with the original
    waveDataCopy = copy.deepcopy(waveData)
    # Shuffle and run FFT_2D in parallel
    num_cores = multiprocessing.cpu_count()
    results = Parallel(n_jobs=num_cores)(
        delayed(run_fft_shuffle)(waveDataCopy, channelIndices, lowerBound, upperBound)
        for _ in range(num_iterations)
    )
    # Average the results
    avgFFT_abs = np.mean([result[0] for result in results], axis=0)
    result = results[1]
    avgMaxAlongvalue = np.mean(result[0]['Max Along Power'])
    avgMaxReversevalue = np.mean(result[0]['Max Reverse Power'])
    avgAlongTempFreq = np.mean(result[0]['Temporal Frequency at Max Along'])
    avgReverseTempFreq = np.mean(result[0]['Temporal Frequency at Max Reverse'])
    avgAlongSpatFreq = np.mean(result[0]['Spatial Frequency at Max Along'])
    avgReverseSpatFreq = np.mean(result[0]['Spatial Frequency at Max Reverse'])
    # Create a new dataframe for averages
    averages_df = pd.DataFrame({
        'Max Along Power': [avgMaxAlongvalue],
        'Max Reverse Power': [avgMaxReversevalue],
        'Temporal Frequency at Max Along': [avgAlongTempFreq],
        'Temporal Frequency at Max Reverse': [avgReverseTempFreq],
        'Spatial Frequency at Max Along': [avgAlongSpatFreq],
        'Spatial Frequency at Max Reverse': [avgReverseSpatFreq]
    })
    # Create new data buckets with shuffled results
    fftBucket_shuffled = wd.DataBucket(avgFFT_abs, "FFT_ABS_shuffled", "trl_spatfreq_tempfreq",chanNames= channelNames)
    waveData.add_data_bucket(fftBucket_shuffled)
    resultBucket_shuffled = wd.DataBucket(averages_df, "2D_FFT_Result_shuffled", 'DataFrame', chanNames= channelNames)
    waveData.add_data_bucket(resultBucket_shuffled)

def run_fft_shuffle(waveData, channelIndices, lowerBound, upperBound):
    np.random.shuffle(channelIndices)  # Shuffle channel indices
    # Call the original FFT_2D function
    FFT_2D(waveData, channelIndices, lowerBound, upperBound)
    # Return the results
    return waveData.get_data('Result'), waveData.get_data('FFT_ABS')

