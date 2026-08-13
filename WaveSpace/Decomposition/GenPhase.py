import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si
from WaveSpace.Utils import WaveData as wd
from WaveSpace.Utils import HelperFuns as hf

# Python translation of:
# https://github.com/mullerlab/generalized-phase
# original matlab code by:
# Lyle Muller (Western University) and Zac Davis (Salk Institute)

#helper funs:
def continous_label(arr):
    changes = np.where(arr[:-1] != arr[1:])[0]  
    return changes

def rewrap(xp):
    return (xp - 2 * np.pi * np.floor((xp - np.pi) / (2 * np.pi)) - 2 * np.pi)

def naninterp(xp):
    nonnan_indices = np.where(~np.isnan(xp))[0]
    xp_nonnan = xp[~np.isnan(xp)]
    nan_indices = np.where(np.isnan(xp))[0]
    return si.pchip_interpolate(nonnan_indices,xp_nonnan,nan_indices)
  	
# main   
def generalized_phase(waveData, dataBucketName = ''):
    """Estimate generalized phase from a broadband signal.

    The algorithm forms an analytic signal using a single-sided Fourier
    representation, detects negative-frequency intervals, and interpolates
    phase across those intervals.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing broadband data.
    dataBucketName : str, default=""
        Name of the input data bucket. By default, the active data bucket is
        used.

    Returns
    -------
    None
        Adds generalized-phase complex data to ``waveData`` as the
        ``complexData`` bucket.

    References
    ----------
    Davis, *Muller, Martinez-Trujillo, Sejnowski, Reynolds. Spontaneous travelling cortical waves gate perception in behaving primates. Nature, 2020 (*equal contribution) 
    """
    reshape = False
    # % parameters
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    newDimord = "trl_chan_time"
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord, newDimord )
    
    nwin = 3
    trials, nChan, npts = currentData.shape
    outcome = np.zeros(currentData.shape, dtype=complex)
    for trialNr in range(trials):
    # % init
        dt = 1 / waveData.get_sample_rate()

        # % analytic signal representation (single-sided Fourier approach, cf. Marple 1999)
        x = currentData[trialNr,:,:].T
        xo = np.fft.fft(x, n=npts, axis=0)
        h = np.zeros(npts)
        if npts > 0 and npts % 2 == 0:
            h[0] = 1
            h[npts // 2] = 1
            h[1:npts//2] = 2
        else:
            h[0] = 1
            h[1:(npts+1)//2] = 2
        #this needs the complex conjugate transpose to do the same as matlab's "'-transpose"
        xo = np.fft.ifft(xo * h[:,np.newaxis], axis=0).conj().T

        #xo = xo.reshape((rows, cols, npts), order='F')
        ph = np.angle(xo)
        md = np.abs(xo)
        
        # calculate IF
        wt = np.zeros(xo.shape)
        wt[:,:-1] = np.angle(xo[:, 1:] * np.conj(xo[:, :-1])) / (2 * np.pi * dt)
        sign_if = np.sign(np.nanmean(wt))
        #% rectify rotation
        if sign_if == -1:
            modulus = np.abs(xo)
            ang = sign_if * np.angle(xo)
            xo = modulus * np.exp(1j * ang)
            ph = np.angle(xo)
            md = np.abs(xo)
            wt[:, :-1] = np.angle(xo[:, 1:] * np.conj(xo[:, :-1])) / (2 * np.pi * dt)
        for ii in range(nChan):
                # check if nan channel
                if np.all(np.isnan(ph[ii, :])):
                    continue
                # find negative frequency epochs 
                idx = (wt[ii, :].squeeze() < 0)
                idx[0] = False
                idxs = continous_label(idx)
                currentbool = True
                #Increase window where negative frequencies are found
                for ind in range(len(idxs)):
                    if currentbool:
                        idx[idxs[ind]+1:idxs[ind]+((idxs[ind+1]-idxs[ind])*nwin)-1] = currentbool
                    currentbool = not currentbool

                p= np.squeeze(ph[ii,:])
                p[idx] = np.nan
                if np.all(np.isnan(p)):
                    continue
                p[~np.isnan(p)] = np.unwrap(p[~np.isnan(p)], axis=0)
                # Interpolate gaps from negative frequencies
                p[np.isnan(p)] = naninterp(p)
                p = rewrap(p)
                ph[ii,:] = p[0:ph.shape[1]]
        #
        outcome[trialNr,:] = md * np.exp( 1j * ph )
        # reshape original data
    dataBucket = wd.DataBucket(outcome, "complexData", newDimord,time= waveData.DataBuckets[waveData.ActiveDataBucket].get_time() ,chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    if hasBeenReshaped:
        dataBucket.reshape(oldshape, currentDimord)  

    waveData.add_data_bucket(dataBucket)  
    
