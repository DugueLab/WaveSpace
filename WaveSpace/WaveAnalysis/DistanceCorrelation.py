import numpy as np
from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import CircStat
from WaveSpace.Utils import WaveData as wa
from WaveSpace.Utils import nsmooth
from WaveSpace.SpatialArrangement import SensorLayout as sensors
import pandas as pd
from pandas import DataFrame
import os
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

def calculate_distance_correlation(waveData, dataBucketName = "", sourcePoints = [], pixelSpacing= 1):
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
    nTrials = complexData.shape[0]
    if os.name == 'posix':  # Unix 
        pool = Pool(cpu_count())
        output = pool.map(distcorr_process_trial, [(ii, complexData, evaluationAngle, tolerance, X, Y, pixelspacing) for ii in range(nTrials)])

    else:  # Windows or Mac
        output = Parallel(n_jobs=cpu_count())(delayed(phase_dist_corr_task)([np.angle(complexData[ii]),ii, sourcePoints, pixelSpacing]) for ii in range(nTrials))
    
    df = pd.concat(output, ignore_index=True)
    if hasBeenReshaped:
        origDimordList = str.split(origDimord, '_')
        groupDims  = [dim for dim in origDimordList if not (dim == "posx" or dim =="posy" or dim == "time")]
        groupDimSizes = origShape[:len(groupDims)]
        multi_indices  = np.array(np.unravel_index(np.arange(complexData.shape[0]), groupDimSizes)).T
          
    phaseCorrBucket = wa.DataBucket(df, "PhaseDistanceCorrelation", "DataFrame", waveData.get_channel_names())
    waveData.add_data_bucket(phaseCorrBucket)

def phase_dist_corr_task(args):
    data, ii, sourcePoints, pixelSpacing, = args
    nTimePoints = data.shape[-1]
    df = DataFrame(columns=['trialInd', 'sourcePointX', 'sourcePointY', 'evaluationPoint', 'rho', 'p'])
    for sourceIndex, sourcePoint in enumerate(sourcePoints):
        for timePoint in range(nTimePoints):
            corr = phase_dist_corr(data[:,:,timePoint], sourcePoint, pixelSpacing)
            df.loc[len(df)] =  ii, sourcePoint[0], sourcePoint[1], timePoint, corr[0], corr[1]
    return df

def calculate_distance_correlation_GP(waveData, dataBucketName = "", evaluationAngle=np.pi, tolerance=0.2):
    """
    Calculate the distance correlation for wave data.

    Python implementation of: https://github.com/mullerlab/generalized-phase.git

    Parameters
    ----------
    waveData : WaveData object
    dataBucketName : str, optional
        Name of the data bucket to use (defaults to the active data bucket). Data in Bucket needs to be complex.
    evaluationAngle : float, optional
        Phase angle (in radians) that triggers the evaluation point (default: np.pi).
    tolerance : float, optional
        Numerical tolerance for phase angle matching (default: 0.2).
    
    Notes
    -----
    Needs a distance matrix to be defined in the WaveData object. See SpatialArrangement if not already there.  
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

    nTrials, nXpos, nYpos, nTime = complexData.shape
    if not np.any(waveData.get_channel_positions()):
        sensors.distmat_to_2d_coordinates_MDS(waveData)
    X = waveData.get_channel_positions()[:, 0]
    Y = waveData.get_channel_positions()[:, 1]
    pixelspacing = distMat[0, 1]
    output = list()

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
        for ind, dim in enumerate(groupDims):
                df.insert(0,dim,0)

        for TargetTrialInd, currentIndex in enumerate(multi_indices):
            indices = df["trialind"] == TargetTrialInd
            for ind, dim in enumerate(groupDims):
                df.loc[indices, dim] = currentIndex[ind]
        df = df.drop(columns = "trialind")
    else:
        df = df.rename(columns={"trialind":"trl"})  
          
    phaseCorrBucket = wa.DataBucket(df, "PhaseDistanceCorrelation", "DataFrame", waveData.get_channel_names())
    waveData.add_data_bucket(phaseCorrBucket)
    

def distcorr_process_trial(args):
    ii, complexData, evaluationAngle, tolerance, X, Y, pixelspacing = args
    complexDataCube = complexData[ii, :, :, :]
    ep = find_evaluation_points(complexDataCube, evaluationAngle, tolerance)
    pm, pd, dx, dy = phase_gradient_complex_multiplication(complexDataCube, pixelspacing)
    source = find_source_points(complexDataCube, X, Y, ep, dx, dy)
    rho = np.zeros((len(ep), 2))
    for idx, thispoint in enumerate(ep):
        ph = np.angle(complexDataCube[:, :, thispoint])
        rho[idx] = phase_dist_corr(ph, source[:, idx], pixelspacing)
    df = DataFrame(data={'trialind': ii, 'rho': rho[:, 0], 'p': rho[:, 1], 'sourcepointsX': source[0],
                     'sourcepointsY': source[1], 'evaluationpoints': ep})

    return df

def phase_dist_corr(ph, source, pixelSpacing):
    """correlation of phase with distance
    (circular-linear), given an input phase map
    INPUT
    ph - phase map (r,c)
    source - source point (sc)
    pixelSpacing - pixel spacing (sc)
    OUTPUT
    cc - circular-linear correlation coefficient, phase correlation w/ distance
    pv - p-value of the correlation (H0: rho == 0, H1: rho != 0)"""
    
    nRows, nColumns = ph.shape
    X = np.meshgrid(np.arange(0,nColumns)-source[1], np.arange(0,nRows)-source[0], indexing='xy')
    D = np.sqrt(X[0]**2 + X[1]**2)
    D = D * pixelSpacing
    D = D.flatten()
    ph = ph.flatten()
    ph[np.isnan(ph)] = None
    D[np.isnan(D)] = None
    cc = np.zeros(2)
    cc[0], cc[1] = CircStat.circular_linear_correlation(ph,D)
    return cc

def phase_gradient_complex_multiplication(complexData, pixel_spacing=1,ifSign=1):
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
    """ep = find_evaluation_points( r, evaluation_angle, tol )**
    estPhase = [ rows X columns X timepoints] complex double of (generalized) phase
    evaluation_angle - angle to evaluate phase crossing [rad]
    tol - numerical tolerance [rad]
    OUTPUT: points at which the phase distribution over channels passes
    a specified angle (within numerical tolerance "tol")"""   
    nRows, nColumns, nTimepoints = complexData.shape
    r = np.reshape(complexData, (nRows*nColumns, nTimepoints))
    r = np.nansum(r, 0) / r.shape[0]
    r = np.abs( CircStat.circular_distance_between_angles(np.angle(r), evaluationAngle))
    dr = (np.where(np.diff(np.sign(np.diff(r)))==2))
    dr= np.array(dr)+1
    ep = dr[0, np.abs(r[dr[0]]) <tolerance]
    return ep

def find_source_points(data, X, Y,evaluationPoints, dx, dy ):
    # % 
    # % FIND SOURCE POINTS     find "putative" source points - the most likely
    # %                          starting point for a wave on the mulichannel
    # %                          array by locating the arg max of divergence of a
    # %                          vector field at each evaluation time point
    # %
    # % INPUT
    # % evaluation_points - time points at which to locate sources
    # % X - x-coordinates for rectangular grid (cf. meshgrid)
    # % Y - y-coordinates for rectangular grid
    # % dx - vector field components along x-direction
    # % dy - vector field components along y-direction
    #     %

    d = np.zeros((data.shape[0], data.shape[1], len(evaluationPoints)))
    d[:,:,:] = np.nan
    for ii, evaluationPoint in enumerate(evaluationPoints):
        d[:,:,ii] = hf.divergence(dx[:,:,evaluationPoint], dy[:,:,evaluationPoint])

    smoothed = nsmooth.smoothn(d,isrobust=True, s=0.2846)
    d = smoothed[0]
    source = np.zeros(( 2, len(evaluationPoints)))
    source[:,:] = np.nan
    for ii in range(len(evaluationPoints)):
        coordinates = np.where( d[:,:,ii] == np.max( d[:,:,ii]))
        if len(coordinates[0]) == 1:
            source[0, ii] = coordinates[0][0]
            source[1, ii] = coordinates[1][0]
    return source