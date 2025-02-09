import numpy as np
from Modules.Utils import HelperFuns as hf
from Modules.Utils import CircStat
from Modules.Utils import WaveData as wa
from Modules.Utils import nsmooth
from Modules.SpatialArrangement import SensorLayout as sensors
import pandas as pd
from pandas import DataFrame
import os
from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed

def calculate_distance_correlation(waveData, dataBucketName = "", evaluationAngle=np.pi, tolerance=0.2):
    #Python implementation of: https://github.com/mullerlab/generalized-phase.git
    if  dataBucketName == "":
        dataBucketName =  waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)

    if not waveData.get_data(dataBucketName).dtype == complex:
        raise TypeError("Data needs to be complex")

    hf.assure_consistency(waveData)
    hf.squareSpatialPositions(waveData)

    if not np.any(waveData.get_distMat()):
        sensors.regularGrid(waveData)
        distMat = waveData.get_distMat()
        print("Warning: No Distance Matrix defined, making regular grid distance matrix on the fly")
    elif waveData.HasRegularLayout:
        distMat = waveData.get_distMat()
        if not sensors.is_regular_grid_2d(distMat):
            print("Warning: Grid not regular")
    else:
        raise RuntimeError("Distance Matrix not found or not regular")

    ComplexPhaseData = waveData.get_data(dataBucketName)
    nTrials, nXpos, nYpos, nTime = ComplexPhaseData.shape

    if not np.any(waveData.get_channel_positions()):
        sensors.distmat_to_2d_coordinates(waveData)
    X = waveData.get_channel_positions()[:, 0]
    Y = waveData.get_channel_positions()[:, 1]
    pixelspacing = distMat[0, 1]
    output = list()

    if os.name == 'posix':  # This is a Unix system
        pool = Pool(cpu_count())
        output = pool.map(distcorr_process_trial, [(ii, ComplexPhaseData, evaluationAngle, tolerance, X, Y, pixelspacing) for ii in range(nTrials)])

    else:  # Windows or Mac
        output = Parallel(n_jobs=cpu_count())(delayed(distcorr_process_trial)(ii, ComplexPhaseData, evaluationAngle, tolerance, X, Y, pixelspacing) for ii in range(nTrials))

    df = pd.concat(output, ignore_index=True)
    phaseCorrBucket = wa.DataBucket(df, "PhaseDistanceCorrelation", "DataFrame", waveData.get_channel_names())
    waveData.add_data_bucket(phaseCorrBucket)


def distcorr_process_trial(args):
    ii, ComplexPhaseData, evaluationAngle, tolerance, X, Y, pixelspacing = args
    ComplexPhaseDataCube = ComplexPhaseData[ii, :, :, :]
    ep = find_evaluation_points(ComplexPhaseDataCube, evaluationAngle, tolerance)
    pm, pd, dx, dy = phase_gradient_complex_multiplication(ComplexPhaseDataCube, pixelspacing)
    source = find_source_points(ComplexPhaseDataCube, X, Y, ep, dx, dy)
    rho = np.zeros((len(ep), 2))
    for idx, thispoint in enumerate(ep):
        ph = np.angle(ComplexPhaseDataCube[:, :, thispoint])
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

def phase_gradient_complex_multiplication(complexPhaseData, pixel_spacing=1,ifSign=1):
    nXpos, nYpos, nTime = complexPhaseData.shape
    dx = np.zeros((nXpos,nYpos,nTime)) 
    dy = np.zeros((nXpos,nYpos,nTime)) 
    for timePoint in range(nTime):
        tmp_dx = np.zeros((nXpos, nYpos))
         # forward differences on left and right edges
        tmp_dx[:,0] = np.angle(complexPhaseData[:,1,timePoint] * np.conj(complexPhaseData[:,0,timePoint])) / pixel_spacing
        tmp_dx[:,nYpos-1] =np.angle(complexPhaseData[:,nYpos-1,timePoint] * np.conj(complexPhaseData[:,nYpos-2,timePoint])) / pixel_spacing
        # centered differences on interior points
        tmp_dx[:,1:nYpos-1] = np.angle(complexPhaseData[:,2:nYpos,timePoint] * np.conj(complexPhaseData[:,0:nYpos-2,timePoint])) / (2*pixel_spacing)
        dx[:,:,timePoint] = tmp_dx * -ifSign

        tmp_dy = np.zeros((nXpos, nYpos))
        tmp_dy[0,:] = np.angle(complexPhaseData[1,:,timePoint] * np.conj(complexPhaseData[0,:,timePoint])) / pixel_spacing
        tmp_dy[nXpos-1,:] =np.angle(complexPhaseData[nXpos-1,:,timePoint] * np.conj(complexPhaseData[nXpos-2,:,timePoint])) / pixel_spacing
        # centered differences on interior points
        tmp_dy[1:nXpos-1,:] = np.angle(complexPhaseData[2:nXpos,:,timePoint] * np.conj(complexPhaseData[0:nXpos-2,:,timePoint])) / (2*pixel_spacing)
        dy[:,:,timePoint] = tmp_dy * -ifSign
    pm = np.sqrt(np.power(dx,2) + np.power(dy,2)) / (2*np.pi)
    pd = np.arctan2(dy, dx)
    return pm, pd, dx, dy

def find_evaluation_points(complexPhaseData, evaluationAngle, tolerance):
    """ep = find_evaluation_points( r, evaluation_angle, tol )**
    estPhase = [ rows X columns X timepoints] complex double of (generalized) phase
    evaluation_angle - angle to evaluate phase crossing [rad]
    tol - numerical tolerance [rad]
    OUTPUT: points at which the phase distribution over channels passes
    a specified angle (within numerical tolerance "tol")"""   
    nRows, nColumns, nTimepoints = complexPhaseData.shape
    r = np.reshape(complexPhaseData, (nRows*nColumns, nTimepoints))
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