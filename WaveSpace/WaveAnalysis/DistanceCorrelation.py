import os
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.linalg import norm
from pandas import DataFrame
from scipy.fftpack.realtransforms import dct, idct

from WaveSpace.SpatialArrangement import SensorLayout as sensors
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wa


def calculate_distance_correlation(waveData, dataBucketName = "", sourcePoints = [], pixelSpacing= 1):
    """Calculate circular-linear phase-distance correlations from source points.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing complex phase data on a two-dimensional
        spatial layout.
    dataBucketName : str, default=""
        Name of the complex input data bucket. By default, the active bucket is
        used.
    sourcePoints : sequence of tuple of int, default=[]
        ``(x, y)`` indices of spatial source points to evaluate.
    pixelSpacing : float, default=1
        Distance between adjacent spatial positions.

    Returns
    -------
    None
        Adds a ``PhaseDistanceCorrelation`` DataFrame bucket to ``waveData``.

    Notes
    -----
    The result stores a correlation coefficient and p-value for every trial,
    source point, and time point.
    """
    if  dataBucketName == "":
        dataBucketName =  waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    hf.assure_consistency(waveData)
    complexData = waveData.get_data(dataBucketName)
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    desiredDimord = "trl_posx_posy_time"
    _hasBeenReshaped, complexData =  hf.force_dimord(complexData, origDimord , desiredDimord)
    nTrials = complexData.shape[0]
    if os.name == 'posix':  # Unix 
        pool = Pool(cpu_count())
        output = pool.map(phase_dist_corr_task, [(np.angle(complexData[ii]),ii, sourcePoints, pixelSpacing) for ii in range(nTrials)])

    else:  # Windows or Mac
        output = Parallel(n_jobs=cpu_count())(delayed(phase_dist_corr_task)([np.angle(complexData[ii]),ii, sourcePoints, pixelSpacing]) for ii in range(nTrials))
    
    df = pd.concat(output, ignore_index=True)
    phaseCorrBucket = wa.DataBucket(df, "PhaseDistanceCorrelation", "DataFrame", sampleRate=waveData.get_sample_rate() ,chanNames= waveData.get_channel_names())
    waveData.add_data_bucket(phaseCorrBucket)

def phase_dist_corr_task(args):
    """
    Parameters
    ----------
    args : tuple

    Returns
    -------
    pandas.DataFrame
    """
    data, ii, sourcePoints, pixelSpacing, = args
    nTimePoints = data.shape[-1]
    df = DataFrame(columns=['trialInd', 'sourcePointX', 'sourcePointY', 'evaluationPoint', 'rho', 'p'])
    for sourcePoint in sourcePoints:
        for timePoint in range(nTimePoints):
            corr = phase_dist_corr(data[:,:,timePoint], sourcePoint, pixelSpacing)
            df.loc[len(df)] =  ii, sourcePoint[0], sourcePoint[1], timePoint, corr[0], corr[1]
    return df

def calculate_distance_correlation_GP(waveData, dataBucketName = "", evaluationAngle=np.pi, tolerance=0.2):
    """Calculate phase-distance correlation using generalized-phase crossings.

    Python implementation of: https://github.com/mullerlab/generalized-phase.git

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object with complex phase data on a regular spatial layout.
    dataBucketName : str, default=""
        Name of the complex input data bucket. By default, the active bucket is
        used.
    evaluationAngle : float, default=numpy.pi
        Mean phase angle in radians used to select evaluation time points.
    tolerance : float, default=0.2
        Numerical tolerance in radians for selecting phase crossings.

    Returns
    -------
    None
        Adds a ``PhaseDistanceCorrelation`` DataFrame bucket containing the
        correlation coefficient, p-value, source point, and evaluation time.
    
    Notes
    -----
    A regular sensor distance matrix must be defined in ``waveData``. See
    :mod:`WaveSpace.SpatialArrangement.SensorLayout`.
    
    References
    -----
    -   Muller L, Piantoni G, Koller D, Cash SS, Halgren E, Sejnowski TJ (2016) Rotating waves during human sleep spindles organize global patterns of activity that repeat precisely through the night. eLife 5: e17267 
    """   
    
    #sanity checks:
    if  dataBucketName == "":
        dataBucketName =  waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    hf.assure_consistency(waveData)

    complexData = waveData.get_data(dataBucketName)
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = complexData.shape
    desiredDimord = "trl_posx_posy_time"
    hasBeenReshaped, complexData =  hf.force_dimord(complexData, origDimord , desiredDimord)

    if not complexData.dtype == complex:
        raise TypeError("Data needs to be complex") 

    if not np.any(waveData.get_distMat()):
        raise TypeError("No Distance Matrix defined. Use SpatialArrangement tools to make one")
    elif waveData.HasRegularLayout:
        distMat = waveData.get_distMat()
        if not sensors.is_regular_grid_2d(distMat):
            print("Warning: Grid not regular")
    else:
        raise RuntimeError("Distance Matrix not found or not regular")

    nTrials, _nXpos, _nYpos, _nTime = complexData.shape
    if not np.any(waveData.get_channel_positions()):
        sensors.distmat_to_2d_coordinates_MDS(waveData)
    X = waveData.get_channel_positions()[:, 0]
    Y = waveData.get_channel_positions()[:, 1]
    pixelspacing = distMat[0, 1]
    output = []

    if os.name == 'posix':  # Unix 
        pool = Pool(cpu_count())
        output = pool.map(distcorr_process_trial, [(ii, complexData, evaluationAngle, tolerance, X, Y, pixelspacing) for ii in range(nTrials)])

    else:  # Windows or Mac
        output = Parallel(n_jobs=cpu_count())(delayed(distcorr_process_trial)([ii, complexData, evaluationAngle, tolerance, X, Y, pixelspacing]) for ii in range(nTrials))

    df = pd.concat(output, ignore_index=True)
    if hasBeenReshaped:
        origDimordList = str.split(origDimord, '_')
        groupDims  = [dim for dim in origDimordList if not (dim == "posx" or dim =="posy" or dim == "time")]
        groupDimSizes = origShape[:len(groupDims)]
        multi_indices  = np.array(np.unravel_index(np.arange(complexData.shape[0]), groupDimSizes)).T
        for dim in groupDims:
                df.insert(0,dim,0)

        for TargetTrialInd, currentIndex in enumerate(multi_indices):
            indices = df["trialind"] == TargetTrialInd
            for ind, dim in enumerate(groupDims):
                df.loc[indices, dim] = currentIndex[ind]
        df = df.drop(columns = "trialind")
    else:
        df = df.rename(columns={"trialind":"trl"})  
          
    phaseCorrBucket = wa.DataBucket(df, "PhaseDistanceCorrelation", "DataFrame", sampleRate=waveData.get_sample_rate() ,chanNames=waveData.get_channel_names())
    waveData.add_data_bucket(phaseCorrBucket)
    

def distcorr_process_trial(args):
    """
    Parameters
    ----------
    args : tuple

    Returns
    -------
    pandas.DataFrame
    """
    ii, complexData, evaluationAngle, tolerance, X, Y, pixelspacing = args
    complexDataCube = complexData[ii, :, :, :]
    ep = find_evaluation_points(complexDataCube, evaluationAngle, tolerance)
    _pm, _pd, dx, dy = phase_gradient_complex_multiplication(complexDataCube, pixelspacing)
    source = find_source_points(complexDataCube, X, Y, ep, dx, dy)
    rho = np.zeros((len(ep), 2))
    for idx, thispoint in enumerate(ep):
        ph = np.angle(complexDataCube[:, :, thispoint])
        rho[idx] = phase_dist_corr(ph, source[:, idx], pixelspacing)
    df = DataFrame(data={'trialind': ii, 'rho': rho[:, 0], 'p': rho[:, 1], 'sourcepointsX': source[0],
                     'sourcepointsY': source[1], 'evaluationpoints': ep})

    return df

def phase_dist_corr(ph, source, pixelSpacing):
    """Calculate circular-linear correlation between phase and source distance.

    Parameters
    ----------
    ph : numpy.ndarray
        Two-dimensional phase map in radians.
    source : tuple of int
        Row and column indices of the source location.
    pixelSpacing : float
        Distance between adjacent spatial samples.

    Returns
    -------
    numpy.ndarray
        Correlation coefficient and two-sided p-value.

    Notes
    -----
    Distances are calculated from ``source`` to every spatial position and
    flattened before calculating the circular-linear correlation.
    """
    
    nRows, nColumns = ph.shape
    X = np.meshgrid(np.arange(0,nColumns)-source[1], np.arange(0,nRows)-source[0], indexing='xy')
    D = np.sqrt(X[0]**2 + X[1]**2)
    D = D * pixelSpacing
    D = D.flatten()
    ph = ph.flatten()
    ph[np.isnan(ph)] = None
    D[np.isnan(D)] = None
    cc = np.zeros(2)
    cc[0], cc[1] = hf.circular_linear_correlation(ph,D)
    return cc

def phase_gradient_complex_multiplication(complexData, pixel_spacing=1,ifSign=1):
    """
    Parameters
    ----------
    complexData : numpy.ndarray
    pixel_spacing : float, default=1
    ifSign : int, default=1

    Returns
    -------
    pm : numpy.ndarray
    pd : numpy.ndarray
    dx : numpy.ndarray
    dy : numpy.ndarray
    """
    nXpos, nYpos, nTime = complexData.shape
    dx = np.zeros((nXpos,nYpos,nTime)) 
    dy = np.zeros((nXpos,nYpos,nTime)) 
    for timePoint in range(nTime):
        tmp_dx = np.zeros((nXpos, nYpos))
         # forward differences on left and right edges
        tmp_dx[:,0] = np.angle(complexData[:,1,timePoint] * np.conj(complexData[:,0,timePoint])) / pixel_spacing
        tmp_dx[:,nYpos-1] =np.angle(complexData[:,nYpos-1,timePoint] * np.conj(complexData[:,nYpos-2,timePoint])) / pixel_spacing
        # centered differences on interior points
        tmp_dx[:,1:nYpos-1] = np.angle(complexData[:,2:nYpos,timePoint] * np.conj(complexData[:,0:nYpos-2,timePoint])) / (2*pixel_spacing)
        dx[:,:,timePoint] = tmp_dx * -ifSign

        tmp_dy = np.zeros((nXpos, nYpos))
        tmp_dy[0,:] = np.angle(complexData[1,:,timePoint] * np.conj(complexData[0,:,timePoint])) / pixel_spacing
        tmp_dy[nXpos-1,:] =np.angle(complexData[nXpos-1,:,timePoint] * np.conj(complexData[nXpos-2,:,timePoint])) / pixel_spacing
        # centered differences on interior points
        tmp_dy[1:nXpos-1,:] = np.angle(complexData[2:nXpos,:,timePoint] * np.conj(complexData[0:nXpos-2,:,timePoint])) / (2*pixel_spacing)
        dy[:,:,timePoint] = tmp_dy * -ifSign
    pm = np.sqrt(np.power(dx,2) + np.power(dy,2)) / (2*np.pi)
    pd = np.arctan2(dy, dx)
    return pm, pd, dx, dy

def find_evaluation_points(complexData, evaluationAngle, tolerance):
    """Find time points at which mean phase crosses a target angle.

    Parameters
    ----------
    complexData : numpy.ndarray
        Complex phase data ordered as rows, columns, and time.
    evaluationAngle : float
        Target phase angle in radians.
    tolerance : float
        Maximum angular difference from the target angle.

    Returns
    -------
    numpy.ndarray
        Indices of qualifying phase-crossing time points.

    Notes
    -----
    The mean complex phase is computed across spatial positions before local
    extrema near ``evaluationAngle`` are selected.
    """
    nRows, nColumns, nTimepoints = complexData.shape
    r = np.reshape(complexData, (nRows*nColumns, nTimepoints))
    r = np.nansum(r, 0) / r.shape[0]
    r = np.abs( hf.circular_distance_between_angles(np.angle(r), evaluationAngle))
    dr = (np.where(np.diff(np.sign(np.diff(r)))==2))
    dr= np.array(dr)+1
    ep = dr[0, np.abs(r[dr[0]]) <tolerance]
    return ep

def find_source_points(data, X, Y,evaluationPoints, dx, dy ):
    """
    Parameters
    ----------
    data : numpy.ndarray
    X, Y : numpy.ndarray
    evaluationPoints : numpy.ndarray
    dx, dy : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    d = np.zeros((data.shape[0], data.shape[1], len(evaluationPoints)))
    d[:,:,:] = np.nan
    for ii, evaluationPoint in enumerate(evaluationPoints):
        d[:,:,ii] = hf.divergence(dx[:,:,evaluationPoint], dy[:,:,evaluationPoint])
    
    d = shortsmooth(d,s=0.2846)
    source = np.zeros(( 2, len(evaluationPoints)))
    source[:,:] = np.nan
    for ii in range(len(evaluationPoints)):
        coordinates = np.where( d[:,:,ii] == np.max( d[:,:,ii]))
        if len(coordinates[0]) == 1:
            source[0, ii] = coordinates[0][0]
            source[1, ii] = coordinates[1][0]
    return source

def shortsmooth(y, s=None):
    """
    Parameters
    ----------
    y : numpy.ndarray
    s : float or None, default=None

    Returns
    -------
    numpy.ndarray
    """
    sizy = y.shape
    W = np.isfinite(y).astype(float)
    isweighted = not np.all(W == 1)

    # Build Lambda 
    Lambda = sum( (2 - 2 * np.cos(np.pi * (np.arange(n) / n))).reshape(
        [n if i == ax else 1 for ax in range(y.ndim)] )
        for i, n in enumerate(sizy) )

    N = np.sum(np.array(sizy) != 1) # tensor rank
    y = np.where(np.isfinite(y), y, 0)
    z = np.zeros_like(y)
    z0 = np.zeros_like(y)
    Wtot = W

    # Relaxation factor
    RF = 1 + 0.75 * isweighted
    xpost = np.array([np.log10(s)])

    for _ in range(2):
        isweighted = True 
        tol, nit = 1, 0 
        while tol>1e-3 and nit<100:
            nit = nit+1
            DCTy = dctND(Wtot*(y-z)+z,f=dct)
            s = 10**xpost[0]
            Gamma = 1./(1+(s*abs(Lambda))**2.0)
            z = RF*dctND(Gamma*DCTy,f=idct) + (1-RF)*z
            tol = isweighted*norm(z0-z)/norm(z)       
            z0 = z 

        # average leverage
        h = np.sqrt(1+16.*s) 
        h = np.sqrt(1+h)/np.sqrt(2)/h 
        h = h**N
        Wtot = W*RobustWeights(y-z, np.isfinite(y),h)
    return z

def RobustWeights(r,I,h):
    """
    Parameters
    ----------
    r : numpy.ndarray
    I : numpy.ndarray
    h : float

    Returns
    -------
    numpy.ndarray
    """
    MAD = np.median(np.abs(r[I]-np.median(r[I]))) # median absolute deviation
    u = np.abs(r/(1.4826*MAD)/np.sqrt(1-h)) # studentized residuals
    c = 4.685 
    W = (1-(u/c)**2)**2.*((u/c)<1) # bisquare weights
    return np.nan_to_num(W)

def dctND(data, f=dct):
    """
    Parameters
    ----------
    data : numpy.ndarray
    f : callable, default=scipy.fftpack.realtransforms.dct

    Returns
    -------
    numpy.ndarray
    """
    for axis in range(data.ndim):
        data = f(data, norm='ortho', type=2, axis=axis)
    return data