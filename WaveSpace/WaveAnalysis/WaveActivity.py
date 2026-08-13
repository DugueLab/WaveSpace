import numpy as np
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd

def find_wave_activity(waveData, dataBucketName=None, dataInd = None, nBases=3):
    """Extract dominant travelling-wave spatial bases from complex data.

    Data are normalized to unit-magnitude phase values before singular-value
    decomposition of their covariance.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing complex-valued phase data.
    dataBucketName : str, default=None
        Name of the input data bucket. By default, the active data bucket is
        used.
    dataInd : tuple of slice, default=None
        Indices selecting a data subset while retaining every input dimension.
        By default, all data are used.
    nBases : int, default=3
        Number of spatial bases to extract.

    Returns
    -------
    None
        Adds ``Bases``, ``Fit``, and ``betas`` data buckets to ``waveData``.
        ``Bases`` contains the spatial patterns, ``Fit`` their per-time-point
        fit, and ``betas`` their complex weights.

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
