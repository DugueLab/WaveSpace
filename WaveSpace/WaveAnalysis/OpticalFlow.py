
import multiprocessing
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from matplotlib import path
from scipy.ndimage import convolve as filter2
from scipy.ndimage import gaussian_filter, generic_filter

from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wd


def create_uv(waveData, applyGaussianBlur=False, type = "real", Sigma=1, alpha = 2, maxIter = 100, is_phase = False, dataBucketName = ''): 
    '''Calculate optical Flow using Horn-Schunck method. Not the fasted pony in the barn....
    
    code partially based on https://github.com/BrainDynamicsUSYD/NeuroPattToolbox (Matlab)
       
    Most of the logic comes from:
    Townsend, R. G., & Gong, P. (2018). 
    Detection and analysis of spatiotemporal patterns in brain 867 activity. 
    PLoS Computational Biology, 14(12), e1006643.

    applyGaussianBlur: if True, apply Gaussian blur to images before calculating optical flow
    Sigma: standard deviation of Gaussian blur kernel
    nIter: number of iterations for Horn-Schunck optical flow calculation
    alpha: smoothness weighting parameter, +alpha => smoother optical flow (typically 0<ALPHA<5) 
    BETA: nonlinear penalty parameter. Beta close to zero will be more accurate, but slow (ToDo: implement!)
    type: 'angle', 'abs' or 'real' (default: 'real') which basis to use for optical flow calculation
    !!careful!!: function expects complex data 
    If you already have angles you want to use, choose type='real' and pass the angles as data, 
    but set is_phase=True!!!
    '''
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
        
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "trl_posx_posy_time")

    _nObs,posx,posy,nframes = currentData.shape
    if type=='angle' and np.iscomplexobj(currentData):
        currentData = np.angle(currentData)
        is_phase = True
    elif type=='abs' and np.iscomplexobj(currentData):
        currentData = np.abs(currentData)
    elif (type=='real') and np.iscomplexobj(currentData):
        currentData = np.real(currentData)        

    # set up initial velocities
    uInitial = np.zeros([posx, posy])
    vInitial = np.zeros([posx, posy])
    # set up averaging kernel for Horn Schunck function
    kernel = np.array([[1 / 12, 1 / 6, 1 / 12],
                       [1 / 6, 0, 1 / 6],
                       [1 / 12, 1 / 6, 1 / 12]], float)

    if os.name == 'posix':  # Linux
        pool = multiprocessing.Pool(multiprocessing.cpu_count()-1)
        result = pool.starmap(uv_process_trial, [(trial_data, nframes , maxIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase) for trial_data in currentData])
        allUV = np.array(result)
        pool.close()
        pool.join()
    else:  # Other OS
        result = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(uv_process_trial)(trial_data, nframes , maxIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase) for trial_data in currentData)
        allUV = np.array(result)

    if hasBeenReshaped:
        #reshape back to original dimord, take into account that the last dimension has been reduced by 1
        allUV = np.reshape(allUV, (*oldshape[:-1], oldshape[-1] - 1)) 
    time = waveData.get_time(dataBucketName)[:-1]      
    dataBucket = wd.DataBucket(allUV, "UV",currentDimord,
                               time=time, 
                               chanNames=waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(dataBucket)

def uv_process_trial(trial_data, nframes , maxIter, uInitial, vInitial, kernel, applyGaussianBlur, Sigma, alpha,is_phase):
    '''not to be called by user directly, but by create_uv.'''
    UV = np.zeros([trial_data.shape[0], trial_data.shape[1], nframes - 1], dtype=complex)
    itersToConverge = []
    for i in range(nframes - 1):
        fn1 = trial_data[:, :, i]
        fn2 = trial_data[:, :, i + 1]
        if applyGaussianBlur:
            if is_phase:#not sure if smoothing makes much sense here in the first plce, but either way, need to do phase data manully
                #because scipy does not handle circular data 
                real_filtered = gaussian_filter(np.real(np.exp(1j * fn1)), Sigma)
                imag_filtered = gaussian_filter(np.imag(np.exp(1j * fn1)), Sigma)
                fn1 = np.angle(real_filtered + 1j * imag_filtered)

                real_filtered = gaussian_filter(np.real(np.exp(1j * fn2)), Sigma)
                imag_filtered = gaussian_filter(np.imag(np.exp(1j * fn2)), Sigma)
                fn2 = np.angle(real_filtered + 1j * imag_filtered)
            else:
                fn1 = gaussian_filter(fn1, Sigma)
                fn2 = gaussian_filter(fn2, Sigma)
        [u, v, iterations] = HS(fn1, fn2, uInitial, vInitial, alpha, kernel, maxIter, is_phase)
        UV[:, :, i] = u + 1j * v
        itersToConverge.append(iterations)
    #print(f"converged in {np.floor(np.mean(itersToConverge))} iterations")
    return UV

def HS(im1, im2, U,V, alpha, kernel, maxIter, is_phase, tol=1e-6):
    """
    Horn - Schunck step
    im1: image at t=0
    im2: image at t=1
    alpha: regularization constant
    Niter: number of iteration
    """
    # derivatives
    [fx, fy, ft] = computeDerivatives(im1, im2, is_phase)
    
    for _iter_count in range(maxIter):
        U_old = U.copy()
        V_old = V.copy()
        
        # Standard HS iteration
        uAvg = filter2(U, kernel)
        vAvg = filter2(V, kernel)
        der = (fx * uAvg + fy * vAvg + ft) / (alpha ** 2 + fx ** 2 + fy ** 2)
        U = uAvg - fx * der
        V = vAvg - fy * der
        
        # Check convergence
        change = np.max(np.abs(U - U_old)) + np.max(np.abs(V - V_old))
        if change < tol:
            break    

    return U, V, _iter_count + 1

def normalize_angle(p):
    return -np.mod(p + np.pi, 2*np.pi) + np.pi

def computeDerivatives(im1, im2, is_phase):
    """get derivatives in x, y and t direction """
    if is_phase:
        # Convert to complex exponentials 
        c1 = np.exp(1j * im1)
        c2 = np.exp(1j * im2)
        
        # spatial gradients using numpy.gradient on complex data
        grad_y_c1, grad_x_c1 = np.gradient(c1)
        grad_y_c2, grad_x_c2 = np.gradient(c2)
        
        # Convert back to phase derivatives
        fx = 0.25 * (np.imag(grad_x_c1 * np.conj(c1)) + np.imag(grad_x_c2 * np.conj(c2)))
        fy = 0.25 * (np.imag(grad_y_c1 * np.conj(c1)) + np.imag(grad_y_c2 * np.conj(c2)))
        
        # Temporal derivative
        ft = normalize_angle(im2 - im1)
        
    else:
        # Regular convolution for non-phase data
        kernelX = np.array([[-1, 1], [-1, 1]]) * .25
        kernelY = np.array([[-1, -1], [1, 1]]) * .25
        kernelT = np.ones((2, 2)) * .25
        
        fx = filter2(im1, kernelX) + filter2(im2, kernelX)
        fy = filter2(im1, kernelY) + filter2(im2, kernelY)
        ft = filter2(im1, kernelT) + filter2(im2, -kernelT)

    return fx, fy, ft
#%%
def poincare_index(uv):
    [row, col] = uv.shape

    SinkSource = np.zeros(uv.shape, dtype='double')
    Saddle = np.zeros(uv.shape, dtype='double')
    d = np.angle(uv)

    # Generic filter here is replacing NLFIlter in matlab as used by Afrashteh 2017.
    # Difference being that we pad the input-array by mirroring. And they don't.
    PoincareIdx = generic_filter(d, P_index1, footprint=np.ones((2, 2)))

    for i in range(row):
        for j in range(col):
            if PoincareIdx[i, j] > 0.9:
                SinkSource[i, j] = 1
            elif PoincareIdx[i, j] < -0.9:
                Saddle[i, j] = 1
    return SinkSource, Saddle

def P_index1(D):
    D = np.reshape(D, (2, 2))
    s = np.zeros(4)
    tap = np.zeros(4)

    s[0] = D[1, 0]
    s[1] = D[1, 1]
    s[2] = D[0, 1]
    s[3] = D[0, 0]

    for i in range(4):
        if i == 2:
            tap[i] = s[3] - s[2]
        else:
            tap[i] = s[np.mod(i + 1, 4)] - s[i]

        if abs(tap[i]) < np.pi / 2:
            tap[i] = tap[i]
        elif tap[i] <= -np.pi / 2:
            tap[i] = tap[i] + np.pi
        else:
            tap[i] = tap[i] - np.pi

    return sum(tap) / np.pi

def SourceSinkSaddle(delta, tau):
    if delta < 0:
        return 0, 0
    if delta == 0:
        return np.nan, np.nan
    # delta > 0
    if tau == 0:
        return 2, 1
    type = 1 if tau > 0 else -1
    if tau * tau < 4 * delta:
        return type, 1
    return type, 0

def makeContours(u, v, Nmin, Lmin_source, Lmax_sink):
    [_uy, ux] = np.gradient(u)
    [vy, _vx] = np.gradient(v)

    div1 = ux + vy  # Divergence
    sourceContours = []
    sinkContours = []
    # [C,h] = contour(div1);
    contour = plt.contour(div1)

    # gets all contours from plot output
    segments = contour.allsegs
    # go through all levels (plt.contour, by default splits input in 10 bins between max and min and draws contours respectively)
    # level is at which height a contour is drawn
    for ind, level in enumerate(contour.levels):
        for currentContour in segments[ind]:
            # check that there is data
            if len(currentContour) > 1:
                # check if contour is
                # a. Long enough
                # b. is deep enough (bigger than minimum)
                # c. Closes (x and y of first equals x and y of last line)
                if len(currentContour) > Nmin and level > Lmin_source \
                        and currentContour[0][0] == currentContour[-1][0] \
                        and currentContour[0][1] == currentContour[-1][1]:
                    sourceContours.append(currentContour)
                # same, but than check if level is lower than max
                if len(currentContour) > Nmin and level < Lmax_sink \
                        and currentContour[0][0] == currentContour[-1][0] \
                        and currentContour[0][1] == currentContour[-1][1]:
                    sinkContours.append(currentContour)
    return sourceContours, sinkContours

def calculate_directional_stability(waveData, dataBucketName = "", windowSize=10):
    """ calculates the directional stability of the UV maps.
        Arguments:

        waveData: waveData object
        dataBucketName: name of the dataBucket to use. Defaults to the last active dataBucket
        windowSize: size of the window in samples during which a vector needs to be pointing in the same direction to be considered stable

        Example: 
        Temporal frequency of interest = 10Hz 
        sampling rate = 250Hz
        WindowSize = 50 samples (2 cycles of the Temporal Frequency) """
        
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    currentData = waveData.get_data(waveData.ActiveDataBucket)
    oldshape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, currentDimord , "trl_posx_posy_time")
    UV_direction = currentData / np.abs(currentData)
    trl, nposx, nposy, nframes = UV_direction.shape
    averageVectors = np.zeros((trl, nposx, nposy, nframes-windowSize), dtype='complex')
    for trialNr in range(trl):
        for frameNr in range(nframes - windowSize): 
            currentUV = UV_direction[trialNr,:, :, frameNr:frameNr + windowSize]
            for x in range(nposx):
                for y in range(nposy):
                    averageVectors[trialNr, x, y, frameNr] = np.sum(currentUV[x, y, :]) / windowSize 
    if hasBeenReshaped:
        #reshape back to original dimord (the -1 in the shape is because the last dimension 
        # has been reduced by the size of the window. passing -1 to reshape makes numpy 
        # figure out the correct size)
        averageVectors = np.reshape(averageVectors, (*oldshape[:-1],-1))
    time = waveData.get_time(dataBucketName)[:-windowSize]  
    dataBucket = wd.DataBucket(averageVectors, "Directional_Stability_Timeseries",waveData.DataBuckets[dataBucketName].get_dimord(), time=time ,chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
    waveData.add_data_bucket(dataBucket)
    
def source_sink_process_trial(thistrialInd, trial_data):
    #initialize arrays
    SourcePoincareJacobian = np.zeros_like(trial_data[:,:,:], dtype=int)
    SinkPoincareJacobian = np.zeros_like(trial_data[:,:,:], dtype=int)
    timepoints = trial_data.shape[2]
    sourceContours = np.empty(timepoints, dtype=object)
    sinkContours = np.empty(timepoints, dtype=object)
    # Initialize dataframes to store confirmed sinks and sources information
    source_df = pd.DataFrame(columns=['trial', 'timepoint', 'posx', 'posy','type'])
    sink_df = pd.DataFrame(columns=['trial', 'timepoint', 'posx', 'posy','type'])
    [uy, ux] = np.gradient(np.real(trial_data[:,:,:]))[:2]
    [vy, vx] = np.gradient(np.imag(trial_data[:,:,:]))[:2]
    for idx in range(timepoints):
        # find critical points
        PoincareSinkSource, PoincareSaddle = poincare_index(trial_data[:, :, idx])
        FixedPointsPoincare = PoincareSinkSource + PoincareSaddle
        [col, row] = (np.where(FixedPointsPoincare.T == 1))
        # Iterate through each detected critical point 
        for f in range(len(row)):
            r = row[f]
            c = col[f]
            # Construct the Jacobian matrix for spatial gradients
            J = np.array([[ux[r, c, idx], uy[r, c, idx]],
                        [vx[r, c, idx], vy[r, c, idx]]])
            # Calculate the determinant and trace of the Jacobian matrix
            delta = np.linalg.det(J)
            tau = np.trace(J)
            # Determine the type of critical point and its stability
            [type, SP] = SourceSinkSaddle(delta, tau)

            if type == 1:
                if SP == 1:
                    SourcePoincareJacobian[r, c, idx] = 1  # for spiral sources
                else:
                    SourcePoincareJacobian[r, c, idx] = 2  # for node sources
            elif type == -1:
                if SP == 1:
                    SinkPoincareJacobian[r, c, idx] = 1  # for spiral sinks
                else:
                    SinkPoincareJacobian[r, c, idx] = 2  # for node sinks
            else:
                SinkPoincareJacobian[r, c, idx] = 0 
                SourcePoincareJacobian[r, c, idx] = 0
                    
    # triple check using gradient of vector field
    # Parameters for source/sink detection
    Nmin = 2  # minimum number of points as the contour size
    Lmin_source = 0.05  # minimum source level
    Lmax_sink = -0.05  # maximum sink level

    for it in range(timepoints):
        u = np.real(trial_data[:, :, it])
        v = np.imag(trial_data[:, :, it])
        sourceContours[it], sinkContours[it] = makeContours(
            u, v, Nmin, Lmin_source, Lmax_sink)
    for ii, potentialSourcePoints in enumerate(np.moveaxis(SourcePoincareJacobian, -1, 0)):
        for thisContour in sourceContours[ii]:
            # Make path from current sourcecontour
            p = path.Path(thisContour)
            # find points
            [rS, cS] = np.nonzero(potentialSourcePoints)
            if (len(rS) > 0):
                coordinates = list(zip(rS, cS))
                # check if points are within contour
                isSource = p.contains_points(coordinates)
                if (np.any(isSource)):
                    for sourceInd, (r, c) in enumerate(coordinates):
                        if isSource[sourceInd]:
                            # Add source info to the dataframe
                            source_df = source_df.append({  
                                'trial': thistrialInd,
                                'timepoint': ii,
                                'posx': r,
                                'posy': c,
                                'type': 'source' 
                            }, ignore_index=True) 

    for ii, potentialSinkPoints in enumerate(np.moveaxis(SinkPoincareJacobian, -1, 0)):
        for thisContour in sinkContours[ii]:
            # Make path from current sourcecontour
            p = path.Path(thisContour)
            # find points
            [rS, cS] = np.nonzero(potentialSinkPoints)
            if (len(rS) > 0):
                coordinates = list(zip(rS, cS))
                # check if points are within contour
                isSink = p.contains_points(coordinates)
                if (np.any(isSink)):
                    for sinkInd, (r, c) in enumerate(coordinates):
                        if isSink[sinkInd]:
                            # Add sink info to the dataframe
                            sink_df = sink_df.append({  
                                'trial': thistrialInd,
                                'timepoint': ii,
                                'posx': r,
                                'posy': c,
                                'type': 'sink'
                            }, ignore_index=True)
    plt.close()
    return source_df, sink_df

def find_sources_sinks(waveData, dataBucketName = ""):
    """requires posx, posy channel dimensions. dataBucketName defaults to "UV" 
    Input: waveData: waveData object
    dataBucketName: name of the dataBucket to use. Defaults to "UV"
    Output: Source and Sink DataFrames with columns: trial, timepoint, posx, posy
    identifies sinks, sources, and saddles in UV maps using the Poincare index theorem 
    and Jacobian matrices. Each critical point is checked against contours of the
    divergence of the vector field to confirm its type and stability.    
    """
    # Find Sources and Sinks
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    UV = waveData.DataBuckets[dataBucketName].get_data()
    currentDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    oldShape = UV.shape
    hasBeenReshaped, UV =  hf.force_dimord(UV, currentDimord , "trl_posx_posy_time")
    _trial, _sizeX, _sizeY, _timepoints = UV.shape

    if os.name == 'posix':  # Linux
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        result = pool.starmap(source_sink_process_trial, [(thistrialInd, trial_data) for thistrialInd, trial_data in enumerate(UV)])        
        pool.close()
        pool.join()
    else:  # Windows or other OS
        from joblib import Parallel, delayed
        num_cores = os.cpu_count()  # Get number of cores
        result = Parallel(n_jobs=num_cores)(delayed(source_sink_process_trial)
                                            (thistrialInd, trial_data) for thistrialInd, trial_data 
                                            in enumerate(UV))

    source_df = pd.concat([x[0] for x in result])
    sink_df = pd.concat([x[1] for x in result])

    source_df = source_df.drop_duplicates()
    sink_df = sink_df.drop_duplicates()

    if hasBeenReshaped:
        def get_original_indices_from_flat(item, oldShape):
            _x, y = oldShape
            x_original = item // y
            y_original = item % y
            return x_original, y_original

        # Iterate through DataFrame entries
        for index, row in sink_df.iterrows():
            # Calculate the original x and y indices
            x_original, y_original = get_original_indices_from_flat(int(row['trial']), oldShape[:2])            
            # Update 'freq' and 'trial' columns
            sink_df.at[index, 'freqBin'] = int(x_original)
            sink_df.at[index, 'trial'] = int(y_original)

        for index, row in source_df.iterrows():
            # Calculate the original x and y indices
            x_original, y_original = get_original_indices_from_flat(int(row['trial']), oldShape[:2])
            
            # Update 'freq' and 'trial' columns
            source_df.at[index, 'freqBin'] = int(x_original)
            source_df.at[index, 'trial'] = int(y_original)

    return source_df, sink_df 

