import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

def find_wave_activity(waveData, dataBucketName=None, dataInd = None, nBases=3):
    """Identifies dominant traveling wave patterns (spatial bases) in complex-valued waveData
    via singular value decomposition on the data covariance. Converts complex data to normalized (unit-length) phase representation.

    Inputs:
    waveData : WaveData object        
    dataBucketName : str, optional
        which data bucket to use
    dataInd : tuple or None, optional
        Optional index tuple to select a subset of the data (e.g., (slice(0,1), slice(10,12), ...)). 
        Needs to result in the same number of dimensions as full data. So use slices for singular dimensions. 
        If None, uses all data.
    nBases : int, optional
        The number of spatial bases to extract (default is 3).

    Adds data buckets to waveData:
    Bases : Complex spatial bases (channels x bases).
    Fit :   fit per trial and time ((freq) trials x time).
    betas : weights ((freq) trials x time x bases).

    References
    ----------
    - https://github.com/ScaleSymmetry/Traveling-wave-analysis https://doi.org/10.1371/journal.pone.0148413
          https://doi.org/10.1371/journal.pcbi.1007316"""
    
    #sanity checks:
    if  dataBucketName == "":
        dataBucketName =  waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)

    hf.assure_consistency(waveData)
    complexData = waveData.get_data(dataBucketName)
    if dataInd:
        complexData = complexData[dataInd]
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = complexData.shape
    desiredDimord = "trl_chan_time"
    hasBeenReshaped, complexData =  hf.force_dimord(complexData, origDimord , desiredDimord)
    # Make complex valued Phase/magnitude Timeseries per freq
    complexData = np.exp(1j*np.angle(complexData))
    #reshape to (trial, time, channel)
    phi= np.transpose(complexData, (0, 2, 1))
    bases, fit, betas = c_TW_bases_betas(phi,nBases=nBases)
    chan_names = waveData.get_channel_names()
    if hasBeenReshaped:
        splitDimensions = origDimord.split("_")
        if "chan" in splitDimensions:
            nGroupDimensions = splitDimensions.index("chan")
            channelShape = origShape[nGroupDimensions]
        elif "posx" in splitDimensions:
            nGroupDimensions = splitDimensions.index("posx")
            channelShape = origShape[nGroupDimensions:nGroupDimensions+2]

        groupDimensions = splitDimensions[0:nGroupDimensions]
        groupDimSizes = origShape[:len(groupDimensions)]
        multi_indices  = np.array(np.unravel_index(np.arange(complexData.shape[0]), groupDimSizes)).T
        
        bases = np.reshape(bases, (*channelShape, bases.shape[-1]))
        basesBucket = wd.DataBucket(bases,"Bases","posx_posy_base",chanNames= chan_names)
        time = waveData.DataBuckets[dataBucketName].get_time()
        fit = np.reshape(fit,(*groupDimSizes, fit.shape[-1]))
        fitBucket = wd.DataBucket(fit,"Fit", ("_").join(groupDimensions) +"_time", time=time ,chanNames=chan_names)
        betas = np.reshape(betas,(*groupDimSizes, betas.shape[-2], betas.shape[-1]))
        betasBucket = wd.DataBucket(betas,"betas",("_").join(groupDimensions) +"_time_beta",time=time ,chanNames= chan_names)
    else:
        basesBucket = wd.DataBucket(bases,"Bases","chan_base", chanNames= chan_names)
        time = waveData.DataBuckets[dataBucketName].get_time()
        fitBucket = wd.DataBucket(fit,"Fit","trl_time", time=time ,chanNames= chan_names)
        betasBucket = wd.DataBucket(betas,"betas","trl_time_beta",time=time ,chanNames= chan_names)
    waveData.add_data_bucket(basesBucket)
    waveData.add_data_bucket(fitBucket)
    waveData.add_data_bucket(betasBucket)

def c_TW_bases_betas(phi_cts,nBases=3):
    #phi complex-valued phase, c cases, t times, s sensors
    phi_Cs = np.asarray(phi_cts.reshape(-1,phi_cts.shape[-1]))
    phi_cent = phi_Cs - phi_Cs.mean(0)
    COV = phi_cent.T.conj()@phi_cent
    u,s,vh = np.linalg.svd(COV)
    bases_sb = vh[:nBases].T
    betas_Cb = phi_cent.dot(bases_sb.conj())
    model_Cs = np.exp(1j*np.angle(bases_sb.dot(betas_Cb.T).T))
    fit_C = (phi_Cs/model_Cs).mean(-1).real
    fit_ct = fit_C.reshape(phi_cts.shape[0],phi_cts.shape[1])
    betas_ctb = betas_Cb.reshape(phi_cts.shape[0],-1,bases_sb.shape[1])
    return bases_sb,fit_ct,betas_ctb
