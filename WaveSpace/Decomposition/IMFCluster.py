import emd    
from scipy import ndimage
from skimage.measure import label, regionprops, regionprops_table  
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wa
import math
import pandas as pd
import copy
import numpy as np
import matplotlib.pyplot as plt 

def cluster_imfs(waveData, highpass, lowpass, dataBucket="complexData", nIMFs=7, plotting = False): 
    emd.logger.set_up()
    emd.logger.set_up(level='CRITICAL')#supress the warning about too few IMFs
    nIMFs=7    
    waveData.set_active_dataBucket(dataBucket)
    hf.assure_consistency(waveData)
    data = waveData.get_data(dataBucket)
    sampleRate = waveData.get_sample_rate()
    _, nTrials, nChannels, nTime = data.shape
    times = np.linspace(0,(data.shape[-1]/sampleRate)-(1/sampleRate),data.shape[-1])
    #make sure histogram doesn't get huge
    if int(np.floor(lowpass-highpass))<30:
        steps= int(np.floor(highpass))
    else:
        steps=30
    freq_edges, freq_centres = emd.spectra.define_hist_bins(highpass+1, lowpass, steps, 'log')
    time_centres = times-((times[2]-times[1])/2)
    # set conditions for temporal and frequency extent of cluster:
    #temporal extent must be minimally ncycles of the centroid frequency
    ncycles = 2
    #to ensure the traveling wave is a well=formed, narrowband oscillation, 
    # the standart deviation of the frequency must be below maxSTD. maxSTD depends on the centroid frequency 
    # either linearly or logarithmically 
    maxSTD_type = 'log' # 'linear'
    ampCutoff = 0.02#cutoff for amplitudes in selectIMF
    #
    complexData = np.empty((nTrials), dtype=object)

    for trl in range(nTrials):
        FreqClustersFound = False
        currentChan = -1
        inst_phase = np.empty((nChannels), dtype=object)
        inst_frequency = np.empty((nChannels), dtype=object)
        inst_amplitude = np.empty((nChannels), dtype=object)
        df= np.empty((nChannels), dtype=object)
        maxAmpind = np.empty((nChannels)).astype(int)

        TW_info = []
        channels = []
        IMF_inds = []
        Cluster_center_frequency = []
        Cluster_start_sample = []  
        Cluster_end_sample =[]
            
        for channel in range(nChannels):
            if trl  == 0 and channel == 0 and plotting:
                plotting = True
            else:
                plotting = False
            signal = data[trl,channel,:]            
            inst_phase[channel], inst_frequency[channel], \
                inst_amplitude[channel] =  ws.EMD_Phase(signal, 
                                                    nIMFs,
                                                    sampleRate,times, 
                                                    plotting)     
            for imfcount in range(inst_frequency[channel].shape[1]):
                if trl  == 0 and channel == 0 and imfcount == 0 and plotting:
                    plotting = True
                else:
                    plotting = False  
                histbincenterperfreq, hht = EMD_HilbertSpec(
                                    inst_frequency[channel][:,imfcount],inst_amplitude[channel][:,imfcount], freq_edges, 
                            freq_centres, time_centres, plotting)
                RegionInds,freq_centroids,RegionStartSample,RegionEndSample = selectIMF(hht,time_centres, freq_centres, ampCutoff,data.info['sfreq'],ncycles, plotting)
                #if such regions are found, add to data frame their trial number, 
                # channel, IMF_index and timefreq extent (coords) 
                if RegionInds:
                    channels.append(channel)
                    IMF_inds.append(imfcount)
                    Cluster_center_frequency.append(freq_centroids)
                    Cluster_start_sample.append(RegionStartSample)#get back to samples
                    Cluster_end_sample.append(RegionEndSample)
                    FreqClustersFound=True# turn this to true as soon as any cluster is found. Only turn back at next trial

        #This will only execute if a potential cluster was found. Put everything that comes out of the channel loop 
        # into a dataframe to make sure channels, frequency and time entries that belong together stay in the same row
        if FreqClustersFound:
            DF= pd.DataFrame({'Channels': channels,'IMF_inds': IMF_inds, 
                            'CenterFreq': Cluster_center_frequency,
                            'Cluster_start_sample': Cluster_start_sample,
                            'Cluster_end_sample': Cluster_end_sample})  
            #find clusters of channels with similar frequency
            potentialTW_info= find_IMF_FreqClusters(DF)
            
            #from here on we use the original IMFs again for precision
            #use potentialTW_info to construct potential traveling wave epochs from the corresponding IMFs
            #initialize arrays for each trial with dimensions [number of potential TWs, channels, time]
            testdata_phase=np.zeros([len(potentialTW_info),nChannels,nTime])
            testdata_phase[:] = np.nan
            testdata_amplitude = copy.deepcopy(testdata_phase)
            testdata_frequency = copy.deepcopy(testdata_phase)
            for ind in range(len(potentialTW_info)): 
                wave_timerange = [potentialTW_info.iloc[ind]['Cluster_start_sample'], potentialTW_info.iloc[ind]['Cluster_end_sample']]
                    
                #now overwrite initialized arrays at the selected timepoints with data  
                for ii,channum in enumerate(potentialTW_info.iloc[ind]['Channels']):
                    testdata_phase[ind,channum,wave_timerange[0]:wave_timerange[1]]=inst_phase[potentialTW_info.iloc[ind]['Channels'][ii]]\
                                            [wave_timerange[0]:wave_timerange[1],\
                                            potentialTW_info.iloc[ind]['IMF_inds'][ii]]
                    testdata_amplitude[ind,channum,wave_timerange[0]:wave_timerange[1]]=inst_amplitude[potentialTW_info.iloc[ind]['Channels']][ii]\
                                                [wave_timerange[0]:wave_timerange[1],\
                                                potentialTW_info.iloc[ind]['IMF_inds'][ii]]
                    testdata_frequency[ind,channum,wave_timerange[0]:wave_timerange[1]]=inst_frequency[potentialTW_info.iloc[ind]['Channels']][ii]\
                                                [wave_timerange[0]:wave_timerange[1],\
                                                potentialTW_info.iloc[ind]['IMF_inds'][ii]]     
                Wavedata = np.multiply(testdata_amplitude[0,10,:], np.cos(testdata_phase[0,10,:])) 
                CurrentcomplexData= testdata_amplitude* np.exp(1j * testdata_phase)
            #build object or whatever this is called with dims [trl][component,channel,time]. 
            # Component is either a potential traveling wave (if clustermethod == overlap) or a singleton dim (if clustermethod == FOI)
            complexData[trl]=CurrentcomplexData
    complexDataBucket = wa.DataBucket(complexData, "IMF_Cluster", "trl_chan_time" )
    waveData.add_data_bucket(complexDataBucket)
    
    #@Seb: here, the dimorder is [trl][component,channel,time], but could also be empty... 'Component' is an intrinsic mode function (so not a fixed frequency) and channels can vary accross trials. We need to keep track of inst_freq for the IMFs that are used in the end


def find_IMF_FreqClusters(DF):
    #DF is a dataframe describing time-frequency regions from individual IMFs.
    #DF attributes for bookkeeping: 
    # channels, IMF indices, center frequency of each continuous time-frequency event, 
    # start and end times of each time-frequency event in samples
    #findFreqClusters finds the time-frequency events that co-occur across multiple channels
    GroupChannels   =[]
    GroupIMF_inds   =[]
    GroupCenterFreq =[]
    GroupStartSample  =[]
    GroupEndSample    =[]
    # make sure each column has only one entry per row
    DF = DF.explode(list(['CenterFreq', 'Cluster_start_sample', 'Cluster_end_sample']))                      
    DF = DF.sort_values('CenterFreq')
    DF['IsPreviousFreqTheSameWithinTolerance'] = DF['CenterFreq'].diff() < (DF['CenterFreq'] / 10) #checks where diff is  within a 10% margin around the Centre freq. Note that this is serial, i.e., the first and last frequency in the series can be much further apart as long as there are intermediate frequencies that meet the condition
    DF['freq_group'] = DF['IsPreviousFreqTheSameWithinTolerance'].ne(DF['IsPreviousFreqTheSameWithinTolerance'].shift()).cumsum()
    DF['freq_group'] = (DF.IsPreviousFreqTheSameWithinTolerance & (DF.IsPreviousFreqTheSameWithinTolerance != DF.IsPreviousFreqTheSameWithinTolerance.shift(1))).cumsum()
    DF.drop(['IsPreviousFreqTheSameWithinTolerance'], axis=1, inplace=True)
    #get median frequency for each group to select matching IMFs later on
    freq_group_means = DF.groupby('freq_group', as_index=False)['CenterFreq'].median()
    #Now go through the frequency groups and check if enough channels contribute
    for groupcount, thisgroup in enumerate(np.unique(DF['freq_group'])):  
        ChannelsInGroup = np.unique(DF[DF['freq_group']==thisgroup]['Channels'].values)   
        #if channels in group is less than 3, ignore that group
        if not len(ChannelsInGroup)<3:                
            TimerangeInGroup = [DF[DF['freq_group']==thisgroup]['Cluster_start_sample'].min(), \
                            DF[DF['freq_group']==thisgroup]['Cluster_end_sample'].max()]            
            #collect all relevant info
            GroupChannels.append(ChannelsInGroup)
            GroupIMF_inds.append(DF[DF['freq_group']==thisgroup]['IMF_inds'].values)
            GroupCenterFreq.append(freq_group_means[freq_group_means['freq_group']==thisgroup]['CenterFreq'].values[0])
            GroupStartSample.append(DF[DF['freq_group']==thisgroup]['Cluster_start_sample'].min())
            GroupEndSample.append(DF[DF['freq_group']==thisgroup]['Cluster_end_sample'].max())

    
    potentialTW = pd.DataFrame({'Channels': GroupChannels,'IMF_inds': GroupIMF_inds, 
                                    'CenterFreq': GroupCenterFreq,
                                    'Cluster_start_sample': GroupStartSample,
                                    'Cluster_end_sample': GroupEndSample}) 
    return potentialTW

def EMD_HilbertSpec(instFreq,instAmp, freq_edges, freq_centres, time_centres, plotting):
    import emd
    import matplotlib.pyplot as plt    
    histbincenterperfreq, hht = emd.spectra.hilberthuang(
    instFreq, instAmp, freq_edges, mode='amplitude', sum_time=False, return_sparse = True)
    if plotting:
        plt.pcolormesh(time_centres, freq_centres, hht, cmap='hot_r', vmin=0)
        plt.colorbar()
    return histbincenterperfreq, hht

def selectIMF(hht,time_centres, freq_centres, ampCutoff,sfreq,ncycles, plotting):
    #selcts only those IMFs that are well formed to be included in further analysis
    #criteria are: amplitude cutoff 

    safetyadd= 0.0001#tiny number to add to denominators to avoid division by 0
    #smooth a bit to allow for slightly mismatched frequencies
    #note that this is only used to detect possible waves, not for
    #estimation. For that we use the higher precision inst_phase
    hht = ndimage.gaussian_filter(hht, 1)
    # img = plt.pcolormesh(time_centres, freq_centres, hht, cmap='hot_r', vmin=0)
    # plt.colorbar()
    hht=np.nan_to_num(hht) #replace nan with 0 so histogram calculation won't fail
    #binarize time-freg
    HilbertSpecBinary=(hht >ampCutoff).astype(int) #remove anything with low amplitude    
    #get properties of Regions with positive values
    Regions = regionprops(label(HilbertSpecBinary))
    Regionprops = regionprops_table(label(HilbertSpecBinary), properties=(
                                    'label','bbox','centroid',
                                        'orientation','area','coords'
                                        ))
    Regionprops =pd.DataFrame(Regionprops)
    RegionLabels=[]#initialyze empty list so checking for entries works later on
    if not Regionprops.empty:
        Regionprops.columns = [c.replace('-', '_') for c in Regionprops.columns]#not really an elegant solution, but cannot filter for column names that contain a '-'...
        #by the time this gets out, there is probably a newer version of skimage that can handle axis scaling, but for now:
        #centroid_0 is the frequency centroid. But regionpropos assumeslinear (pixel-spaced) axes. To translate to frequency bins, we index ino the freq-centres vector
        Regionprops['centroid_0'] = freq_centres[np.ndarray.round(Regionprops['centroid_0'].values).astype(int)]
        #same for time:
        Regionprops['timeStart'] = (time_centres[np.ndarray.round(Regionprops['bbox_1'].values).astype(int)])*1000
        Regionprops['timeEnd'] = (time_centres[np.ndarray.round(Regionprops['bbox_3'].values).astype(int)-1])*1000
        #...and freq bounding box
        Regionprops['bbox_0'] = freq_centres[np.ndarray.round(Regionprops['bbox_0'].values).astype(int)]
        Regionprops['bbox_2'] = freq_centres[np.ndarray.round(Regionprops['bbox_2'].values).astype(int)-1]
        #filter for clusters that are at least 2 cycles long and do not vary too much in frequency
        Regionprops = Regionprops[(Regionprops.timeEnd - Regionprops.timeStart >=ncycles*(sfreq/Regionprops.centroid_0+safetyadd)) \
                        & (Regionprops.bbox_2 - Regionprops.bbox_0 <= 5*(np.log(Regionprops.centroid_0+safetyadd)))]
        RegionLabels = list(Regionprops.label) 

        if plotting and RegionLabels:        
            fig, ax = plt.subplots(figsize= [25,35] )
            ax.imshow(HilbertSpecBinary, cmap=plt.cm.gray, origin ='lower')
            for props in Regions:
                y0, x0 = props.centroid
                orientation = props.orientation
                x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)
                ax.plot((x0, x1), (y0, y1), '-g', linewidth=2.5)
                ax.plot((x0, x2), (y0, y2), '-g', linewidth=2.5)
                ax.plot(x0, y0, '.g', markersize=15)
                ax.plot(bx, by, '-b', linewidth=2.5)        
            plt.show()             
            #display without all the coordinates for easier overview
            #display(Regionprops[Regionprops.columns.difference(['coords'])]) 