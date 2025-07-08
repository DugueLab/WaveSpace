import numpy as np
from sklearn import cluster
import matplotlib.pyplot as plt
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Decomposition.Hilbert as h

def get_frequency_cluster(waveData, dataBucket="", freqList=[]):
    hf.assure_consistency(waveData)

    if dataBucket == "":
        data = waveData.get_data(waveData.ActiveDataBucket)
    else:
        data = waveData.get_data(dataBucket)
        waveData.set_active_dataBucket(dataBucket)
    sampleRate = waveData.get_sample_rate()
    nTrials, nChannels, nTime = data.shape

    complexData_FreqCluster = np.empty(complexData.shape[0], dtype=object)
    ClusterContacts=np.empty(complexData.shape[0], dtype=object)
    
    #take power-spectra, subtract 1/f, find freq peaks, cluster channels that have freq peaks in common
    for trl in range(complexData.shape[0]):        
        power = np.mean(abs(complexData[trl]) **2, axis=2).T
        # power_log = np.log10(abs(complexData[trl]) **2)
        # power_log_mean = (np.mean(power_log, axis = 1)).T
        FreqPeaks = find_freq_peaks(power, freqList, Options["plotting"])
        #find clusters of channels with similar freq-peaks
        maxDist = None
        FreqPeaks = np.delete(FreqPeaks, [1,2] , 1)#remove unnecessary columns. Each row of FreqPeaks is now peak, channel
        FreqClusterContacts, FreqClusterFrequencies = find_freq_cluster(complexData,FreqPeaks, distMat, maxDist)
        #Note that there can be several Clusters per (median) frequency. 
        #This happens if there are several spatially separate groups of contacts with a similar FreqPeak

        #now for each cluster, filter around the cluster freq peak and do Hilbert transform
        temp = np.empty(len(FreqClusterFrequencies), dtype=object)
        for clusterind in range(len(FreqClusterFrequencies)):
            freq = [FreqClusterFrequencies[clusterind] *.90, FreqClusterFrequencies[clusterind]/.90]#Cluster median frequency +- 10%
            data_to_filter = copy.deepcopy(data)#use the raw data here so no filters (and data cropping) have been applied 
            #@Seb:this needs to change to make sure other steps have happened. Maybe use the data_preproc.previous data option here in some way???
            #pick current trial
            data_to_filter = data_to_filter._getitem(trl)
            #pick channels in current cluster
            data_to_filter.pick(FreqClusterContacts[clusterind])
            NarrowbandData = ws.narrowbandFilter(data_to_filter,
                                    LowCutOff  = freq[0], 
                                    HighCutOff = freq[1],
                                    plotting = False)
            NarrowbandData.crop(t0,t1)
            data_hilbert = h.apply_hilbert(NarrowbandData, False)
            inst_amplitude = np.abs(data_hilbert)
            inst_phase = np.angle(data_hilbert)
            temp[clusterind]= np.squeeze(inst_amplitude * np.exp(1j * inst_phase))
        complexData_FreqCluster[trl]=temp
        ClusterContacts[trl]=FreqClusterContacts
    #Overwrite complexData with new one:
    complexData = complexData_FreqCluster

def find_freq_cluster(complexData,FreqPeaks,distMat,maxDist=None):
    Cluster = []
    #define neighbors from channel distances. Note that this depends on the datatype. 
    #e.g.,EEG allows for larger distances than ECoG
    plotting = True #@Seb: needs to change to new syle
    if not maxDist:
        maxDist = 2*(np.min(distMat[np.nonzero(distMat)]))
    #find frequency clusters (peak freq + some tolerance)
    x = copy.deepcopy(FreqPeaks)
    x[:,1]=0    
    bandwidth = cluster.estimate_bandwidth(x, quantile=0.1)
    MeanShift = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    MeanShift.fit(x)
    freq_clust_labels = MeanShift.labels_
    unique_freq_clust_labels = np.unique(freq_clust_labels)
    n_freq_clusters_ = len(unique_freq_clust_labels)
    colors = plt.cm.nipy_spectral(np.linspace(0, 1, n_freq_clusters_))
    ClusterFreq=[]
    if plotting:
        plt.figure()
        for this_freq_cluster in range(n_freq_clusters_):
            contact_included_in_this_freq_clust = freq_clust_labels == this_freq_cluster
            plt.scatter(FreqPeaks[contact_included_in_this_freq_clust,1],FreqPeaks[contact_included_in_this_freq_clust,0],c=colors[this_freq_cluster,None], label = str(this_freq_cluster))        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title('freqpeaks per channel. Color is freq-cluster')
        plt.show()
    for this_freq_cluster in range(n_freq_clusters_):
        contact_included_in_this_freq_clust = freq_clust_labels == this_freq_cluster        
        contactIDs_in_this_freqcluster = np.unique(FreqPeaks[contact_included_in_this_freq_clust,:][:,1].astype(int))#the unique is there because a signle contact can appear in a cluster twice if it has to freq peaks close together
        clustermedianfreq=np.median(FreqPeaks[contact_included_in_this_freq_clust,:][:,0])
        #mask freq-cluster result with distance matrix to exclude hannels that have similar frequency, 
        #but are not spatially contiguous 
        clustering= cluster.DBSCAN(eps=maxDist,metric = 'precomputed', min_samples=1).fit(distMat[contactIDs_in_this_freqcluster[:,None],contactIDs_in_this_freqcluster[None,:]])
        Spatial_clust_labels= clustering.labels_
        for this_spatial_cluster in np.unique(Spatial_clust_labels):
            contact_in_spatial_cluster= Spatial_clust_labels==this_spatial_cluster
            #check if spatial cluster contains enough contacts to be considered (default=3)
            #if so, add IDs of contacts in cluster and cluster median frequency to respective lists
            if np.sum(contact_in_spatial_cluster)>2:           
               Cluster.append(contactIDs_in_this_freqcluster[contact_in_spatial_cluster]) 
               ClusterFreq.append(clustermedianfreq)
        # if plotting:
        #     plt.figure()
        #     plt.scatter(distMat[contactIDs_in_this_freqcluster,0],distMat[contactIDs_in_this_freqcluster,1], c= Spatial_clust_labels)
        #     plt.title('distance between contacts within a freq-cluster. Color is spatial cluster')
        #     plt.show()
    return Cluster, ClusterFreq

def find_freq_peaks(power, freqlist, plotting):
    #find relative freq peak in power spectrum (with 1/f removed)
    group = FOOOFGroup(verbose=False)

    # Fit the spectral parameterization model
    group.fit(freqlist, power)
    peaks = group.get_params('peak_params')
    if plotting:
        group.plot()
        # Plot example spectrum
        fm = group.get_fooof(ind=0, regenerate=True)

        # Print results and plot extracted model fit
        fm.print_results()
        fm.plot()
    return peaks