import copy
import time
from math import sqrt

import numpy as np
import plotly.graph_objects as go
import scipy.stats
from pint import UnitRegistry

from WaveSpace.Utils import WaveData as wd


def circular_distance_between_angles(angle1, angle2):
    """
    Parameters
    ----------
    angle1, angle2 : numpy.ndarray or float

    Returns
    -------
    numpy.ndarray or float
    """
    return np.pi -np.abs(np.pi - np.abs(angle1 - angle2))

def circular_linear_correlation(alpha, x):
    """
    Parameters
    ----------
    alpha, x : numpy.ndarray

    Returns
    -------
    rho : float
    pval : float
    """
    n = np.size(alpha)
    rxs = np.corrcoef(x, np.sin(alpha))[0,1]
    rxc = np.corrcoef(x, np.cos(alpha))[0,1]
    rcs = np.corrcoef(np.sin(alpha), np.cos(alpha))[0,1]
    rho = np.sqrt((rxc**2 + rxs**2 - 2*rxc*rxs*rcs)/(1-rcs**2))
    pval = 1-scipy.stats.chi2.cdf(n*rho**2,2)
    return rho, pval


def mean2(x):
    """
    Mean 2 like in Matlab

    Parameters
    ----------
    x : numpy.ndarray

    Returns
    -------
    float
    """
    return np.sum(x) / np.size(x)

def divergence(U, V):
    """Compute divergence of a two-dimensional vector field.

    Parameters
    ----------
    U, V : numpy.ndarray
        Horizontal and vertical vector-field components.

    Returns
    -------
    numpy.ndarray
        Divergence at each spatial position.
    """
    _dud, px = np.gradient(U)
    qy, _dud = np.gradient(V)
    return px+qy

def add_noise_channels(data, proportion=1.0, No_noise_channels=10):
    """Add noise channels to the data with a variance set to a proportion of the input data's variance.

    Parameters
    ----------
    data : numpy.ndarray
        Data ordered as trials, channels, and time.
    proportion : float, default=1.0
        Multiplier applied to the mean within-trial channel variance.
    No_noise_channels : int, default=10
        Number of noise channels appended to each trial.

    Returns
    -------
    numpy.ndarray
        Input data with appended noise channels.
    """

    # Get data dimensions
    trials, channels, nt = data.shape

    # Prepare the data_out with added noise channels
    data_out = np.zeros((trials, channels + No_noise_channels, nt))

    for t in range(trials):
        # Calculate Noise power for current trial
        Noise_power = np.mean(np.var(data[t], axis=1)) * proportion

        # Add the data to data_out
        data_out[t, :channels] = data[t]

        # Generate noise and add to data_out
        for i in range(No_noise_channels):
            data_out[t, channels + i] = np.random.randn(nt) * np.sqrt(Noise_power)

    return data_out

def find_nearest(array, value):
    """position and value of element in array closest to value.

    Parameters
    ----------
    array : array-like
        Values to search.
    value : float
        Target value.

    Returns
    -------
    ind : int
        Index of the closest value.
    nearest_value : float
        Array value at ``ind``.
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    ind = np.nanargmin(np.abs(array - value))
    return ind, array[ind]

def find_max_signal(data, dimension, averageDimension=None):
    """Return index of signal with highest amplitude, averaged over averageDimension

    Parameters
    ----------
    data : numpy.ndarray
        Input data.
    dimension : int
        Axis along which to locate the maximum.
    averageDimension : int or None, default=None
        Axis averaged before finding the maximum.

    Returns
    -------
    ind : numpy.ndarray or int
        Index or indices of the maximum signal.
    value : float
        Mean value at the selected index or indices.
    """
    # Average over dimension
    avg = np.nanmean(data, axis=averageDimension)
    # Find index of maximum
    ind = np.nanargmax(avg, axis=dimension)
    return ind, np.nanmean(avg[ind])

def scale(x, out_range=(-1, 1), axis=None):
    """Scale any input to be between out_range[0] and out_range[1]; defaults to range (-1 1)

    Parameters
    ----------
    x : numpy.ndarray
    out_range : tuple of float, default=(-1, 1)
    axis : int or None, default=None

    Returns
    -------
    numpy.ndarray
    """
    
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def square_rooted(x):
    """
    Parameters
    ----------
    x : array-like

    Returns
    -------
    float
    """
    return round(sqrt(sum([a * a for a in x])), 3)

def cartesian_product_trials(**arrays):
    """
    Parameters
    ----------
    **arrays : array-like

    Returns
    -------
    dict of numpy.ndarray
    """
    out_dict = {}
    for key, value in arrays.items():
        if isinstance(value, (int, float)):  # check if value is a single number
            out_dict[key] = np.array([value])
        elif isinstance(value, np.ndarray) and value.ndim == 1:  # check if value is a 1D array
            out_dict[key] = value
        else:
            out_dict[key] = np.asarray(value).flatten()
    la = len(out_dict)
    dtype = np.result_type(*out_dict.values())
    arr = np.empty([len(a) for a in out_dict.values()] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*out_dict.values())):
        arr[...,i] = a
    arr = arr.reshape(-1, la)
    for i, key in enumerate(out_dict.keys()):
        out_dict[key] = arr[..., i].flatten()
    return out_dict

def cosine_similarity(x, y):
    """
    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    float
    """
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    return round(numerator / float(denominator), 3)

def posxy_to_chan(data):
    """
    Parameters
    ----------
    data : numpy.ndarray

    Returns
    -------
    data : numpy.ndarray
    shape : tuple of int
    """
    shape = data.shape
    chanDim = shape[-3] * shape[-2]
    newShape = (*shape[0:-3], chanDim, shape[-1])
    data = np.reshape(data, newShape)
    return data, shape

def force_dimord(data,  currentDimord, desiredDimord):
    """Reshape data to satisfy a requested dimension order.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array.
    currentDimord, desiredDimord : str
        Current and requested underscore-separated dimension orders.

    Returns
    -------
    hasBeenReshaped : bool
        Whether the data shape was changed.
    data : numpy.ndarray
        Data reshaped to match ``desiredDimord`` where possible.

    Notes
    -----
    Two spatial dimensions can be flattened into a channel dimension. This is
    an internal bookkeeping helper not meant to be called directly by the user.
    """
    if (currentDimord == desiredDimord):
        return False, data
    
    newDims = desiredDimord.split("_")
    oldDims = currentDimord.split("_")
    
    last_dims = oldDims[-3:]
    if last_dims[0] == 'posx':
        last_dims.insert(0, "trl")

    new_last_dims = newDims[-3:]
    if new_last_dims[0] == 'posx':
        new_last_dims.insert(0, "trl")

    chanshape = ()
    if new_last_dims != last_dims: 
        if (len(new_last_dims) > len(last_dims)): 
            raise ValueError("Function requires XY positions")
        if (len(new_last_dims) < len(last_dims)): 
            data, chanshape = posxy_to_chan(data)
            oldDims.remove("posx")
            oldDims.remove("posy")
            oldDims.insert(-1 ,"chan")
            last_dims = new_last_dims

    if (len(newDims) == len(oldDims)):  
        if chanshape:
            return True, data
        else:
            return False, data
    else:
        groupShape = data.shape[:-(len(last_dims))]
        lastDimShape = data.shape[-(len(last_dims)):]
        newshape = data.shape 
        while (len(newDims) > len(oldDims)):
            oldDims.insert(0, "new_dim")
            newshape = (1, *newshape)
        if (len(newDims) < len(oldDims)):
            currentGroupDims = list(set(oldDims) - set(last_dims))
            if len(currentGroupDims) > 0:                
                groupsizeproduct = 1
                for ii in range(len(currentGroupDims)):
                    groupsizeproduct = groupsizeproduct * groupShape[ii]
                newgroupshape = groupsizeproduct * lastDimShape[0]
                newshape = (newgroupshape , *lastDimShape[1:])
    data = data.reshape(newshape)
    assert len(data.shape) == len(desiredDimord.split("_")), "Something is wrong with your data-dimensions, cannot reshape to required dimord"
    return True, data

def combine_grad_sensors(waveData, dataBucketName="", method='RMS'):
    """Combine ordered pairs of gradiometer channels.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing paired gradiometer channels.
    dataBucketName : str, default=""
        Input data bucket. An empty string uses the active bucket.
    method : {"RMS", "mean"}, default="RMS"
        Method used to combine each channel pair.

    Returns
    -------
    None
        Adds combined, x-gradient, and y-gradient data buckets to ``waveData``.

    Notes
    -----
    Channel pairs must be consecutive, and channel-position metadata is
    reduced to the first channel of each pair. CAUTION: assumes that all channels are gradiometers, with pairs listed in order.
    """
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    assure_consistency(waveData)

    data=waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    originalDimord = waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    originalShape = data.shape
    if originalDimord != "trl_chan_time":
        has_been_reshaped, data = force_dimord(data, originalDimord, "trl_chan_time")
    else:
        has_been_reshaped = False

    data = data.reshape(data.shape[0], data.shape[1] // 2, 2, data.shape[2])
    gradXData = data[:,:,0,:]
    gradYData = data[:,:,1,:]
    if method == 'RMS':
        combined_data = np.sqrt(np.sum(data**2, axis=2))
    elif method == 'mean':
        combined_data = np.mean(data, axis=2)
    else:
        raise ValueError("Invalid method. Method must be 'RMS' or 'mean'.")
    newChanNames = ['CP_' + name for name in waveData.get_channel_names()[::2]]

    if has_been_reshaped:
        # Calculate the new shape
        new_shape = originalShape[:-2] + combined_data.shape[-2:]
        # Reshape the combined data to the new shape
        combined_data = combined_data.reshape(new_shape)
        gradXData = gradXData.reshape(new_shape)
        gradYData = gradYData.reshape(new_shape)
    
    # Create a new DataBucket with the combined sensor data
    combined_dataBucket = wd.DataBucket(
        combined_data, 
        dataBucketName + "_combinedPlanar", 
        waveData.DataBuckets[dataBucketName].get_dimord(),
        time=waveData.DataBuckets[dataBucketName].get_time(),
        chanNames= newChanNames)
    
    GradX_dataBucket = wd.DataBucket(
        gradXData, 
        dataBucketName + "_GradX", 
        waveData.DataBuckets[dataBucketName].get_dimord(), 
        time=waveData.DataBuckets[dataBucketName].get_time(),
        chanNames=newChanNames)
    
    GradY_dataBucket = wd.DataBucket(
        gradYData, 
        dataBucketName + "_GradY", 
        waveData.DataBuckets[dataBucketName].get_dimord(),
        time=waveData.DataBuckets[dataBucketName].get_time() ,
        chanNames= newChanNames)
    
    waveData.set_channel_positions(waveData.get_channel_positions()[::2])#remove the second channels of each pair from positions
    waveData.add_data_bucket(combined_dataBucket)
    waveData.add_data_bucket(GradX_dataBucket)
    waveData.add_data_bucket(GradY_dataBucket)
    waveData.log_history(["combine_grad", f"combined gradiometers using {method}"])


def mutual_information(hgram):
    """Calculate mutual information from a joint histogram.

    Parameters
    ----------
    hgram : numpy.ndarray
        Joint histogram counts.

    Returns
    -------
    float
        Mutual information in natural-log units.
    """
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1)  # marginal for x over y
    py = np.sum(pxy, axis=0)  # marginal for y over x
    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # do calculation using pxy, px_py 2D arrays
    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def order_to_grid(data,shape,dimord):
    """
    Reshapes the active databucket of a WaveData object into a grid with the specified shape and dimension order.  

    Parameters
    ----------
    data : WaveSpace.Utils.WaveData.WaveData
        WaveData object
    shape : tuple of int
        Shape of desired grid 
    dimord : str
        Underscore seperated dimension order like ``trl_posx_posy_time``.
    Returns
    -------
    None
        Changes active databucket shape in place for 

    """
    
    dimord = dimord.split('_')
    for ind, dimension in enumerate(dimord):
        if dimension == "chan":
            chandim = ind

    oldshape = np.asarray(data.DataBuckets[data.ActiveDataBucket].get_data().shape)
    newShape = np.hstack((oldshape[0:chandim], np.asarray(shape),oldshape[chandim+1:]))
    newShape = tuple(newShape)
    dimord[chandim] = "posx_posy"
    data.DataBuckets[data.ActiveDataBucket].reshape(newShape,'_'.join(dimord))
    data.log_history(["order_to_grid", "order", "Square"])

def assure_consistency(waveData, dataBucketName = None):
    """
    Ensure dimensions of data in <dataBucketName> are consistent with its dimord. If the data is missing a trial dimension, add one of size 1. 

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        waveData object
    dataBucketName : str or None, default=None

    Returns
    -------
    None
    """
    if dataBucketName == None:
        dataBucketName = waveData.ActiveDataBucket
    dimord = waveData.DataBuckets[dataBucketName].get_dimord()
    shape = waveData.get_data(dataBucketName).shape
    dimensions = dimord.split("_")
    if "trl" not in dimensions and dimensions[-1] == "time":
        data = waveData.DataBuckets[dataBucketName].get_data()
        if dimensions[-2] == "posy":
            dimensions.insert(-3,"trl")
            data = np.reshape(data, (*shape[:-3] ,1 , *shape[-3:])) 
            shape = data.shape
        elif dimensions[-2] == "chan":
            dimensions.insert(-2,"trl")
            data = np.reshape(data, (*shape[:-2] ,1 , *shape[-2:])) 
            shape = data.shape
        waveData.DataBuckets[dataBucketName].set_data(data, "_".join(dimensions) )
        dimord = waveData.DataBuckets[dataBucketName].get_dimord()
        print("added trial-dimension of size 1")
    assert len(shape) == len(dimensions), "Dimensions of inputData inconsistent with Dimord"
    assert dimord[-13:] == "trl_chan_time" or dimord[-18:] == "trl_posx_posy_time", "dimord does not contain trl_chan_time or trl_posx_posy_time in the appropriate positions"

def convert_units(dataBucket, new_unit):
    """
    Parameters
    ----------
    dataBucket : WaveSpace.Utils.WaveData.DataBucket
    new_unit : str

    Returns
    -------
    None
    """
    ureg = UnitRegistry()
    # Check if the new unit is valid and the conversion is possible
    try:
        quantity = 1 * ureg(dataBucket._unit)
        quantity.to(new_unit)
    except Exception:
        print(f"Warning: Conversion from {dataBucket._unit} to {new_unit} is not supported.")
        return

    # Perform the conversion and modify the dataBucket in place
    dataBucket._data = np.array([(d * ureg(dataBucket._unit)).to(new_unit).magnitude for d in dataBucket._data])
    dataBucket._unit = new_unit

def plot_chanpos(waveData, show_names = True):
    """
    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
    show_names : bool, default=True

    Returns
    -------
    None
    """
    chanpos = waveData.get_channel_positions()
    chan_names = waveData.get_channel_names()
    fig = go.Figure(data=[go.Scatter3d(
        x=chanpos[:, 0],
        y=chanpos[:, 1],
        z=chanpos[:, 2],
        mode='markers',
        text=chan_names if show_names else None,
        marker={
            "size": 6,
            "color": 'blue',
        }
    )])

    # Add title and labels
    fig.update_layout(
        scene={
            "xaxis_title": 'X',
            "yaxis_title": 'Y',
            "zaxis_title": 'Z'
        },
        title="Channel Positions",
    )

    fig.show()
         
def squareSpatialPositions(waveData):
    """Reshape active databucket into two spatial dimensions.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object whose active bucket has a square number of channels.

    Returns
    -------
    None
        Updates the active bucket's data shape and dimension order in place.

    Notes
    -----
    The ``chan`` dimension is replaced with ``posx_posy`` using the square
    root of the channel count for both spatial axes. Only use when you are sure this is appropriate 
    """
    dimord = waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    shape = waveData.get_active_data().shape
    dimensions = dimord.split("_")
    if len(dimensions) ==3:
        for ind, dimension in enumerate(dimensions):
            if dimension == "chan":
                chandim = ind
        oldshape = np.asarray(shape)
        addedShape = ((int(np.sqrt(oldshape[chandim])),int(np.sqrt(oldshape[chandim]))))
        newShape = np.hstack((oldshape[0:chandim], np.asarray(addedShape),oldshape[chandim+1:]))
        dimensions[chandim] = "posx_posy"
        waveData.DataBuckets[waveData.ActiveDataBucket].reshape(newShape,'_'.join(dimensions))

def tic():
    """
    Returns
    -------
    float
    """
    #Homemade version of matlab tic and toc functions
    startTime_for_tictoc = time.time()
    return startTime_for_tictoc

def toc(startTime_for_tictoc):
    """
    Parameters
    ----------
    startTime_for_tictoc : float

    Returns
    -------
    float
    """
    toctime = time.time() - startTime_for_tictoc
    return toctime

def relative_phase(waveData, ref=None, dataBucketName=""):
    """Calculate phase relative to a reference channel or position.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing complex-valued data.
    ref : tuple of int, int, or numpy.ndarray, default=None
        Spatial ``(posx, posy)`` reference, channel index, or reference time
        series.
    dataBucketName : str, default=""
        Input data bucket. An empty string uses the active bucket.



    Returns
    -------
    None
        Adds relative phase in radians to ``waveData`` as the
        ``relativePhase`` bucket.

    Raises
    ------
        ValueError: If neither ref_coord nor refchan_ind are provided, or if both are provided.
        ValueError: If the dimensions of the data do not match the type of reference provided (i.e., refchan_ind with 4D data or ref_coord with 3D data).

    Notes
    -----
        Negative values mean lag, positive values mean lead. All values are expressed in radians.
    """
    # Ensure proper bookkeeping of data dimensions
    if dataBucketName == "":
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    # Retrieve the complex data from the active data bucket
    complex_data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    origShape = complex_data.shape
    original_dimord = waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    
    if "posx_posy" in original_dimord:
        has_been_reshaped, complex_data = force_dimord(complex_data, original_dimord, "trl_posx_posy_time")
    else:
        has_been_reshaped, complex_data = force_dimord(complex_data, original_dimord, "trl_chan_time")

    if ref is None:
        raise ValueError('You must provide a reference coordinate, channel index or reference timeseries.')

    # Select the reference data
    if isinstance(ref, tuple):
        ref_data = complex_data[:, ref[0], ref[1], :]
        ref_data = ref_data[:, np.newaxis, np.newaxis, :]
    elif isinstance(ref, int):
        ref_data = complex_data[:, ref, :]
        ref_data = ref_data[:, np.newaxis, :]
    elif isinstance(ref, np.ndarray):
        ref_data = ref
        ref_data = ref_data[np.newaxis, np.newaxis, :]
    else:
        raise TypeError('ref must be a tuple, an integer, or a numpy array.')

    # Calculate relative phase using complex division
    rel_complex_data = complex_data / ref_data

    # Calculate the phase angle of the relative complex data
    rel_phase = np.angle(rel_complex_data)

    if has_been_reshaped:
        # Reshape back to original dimord
        rel_phase = np.reshape(rel_phase, origShape)
        phase_dimord = original_dimord
    else:
        phase_dimord = "trl_chan_time" if isinstance(ref, int) else "trl_posx_posy_time"

    # Create a new DataBucket with relative phase
    PhaseDataBucket = wd.DataBucket(rel_phase, "relativePhase", phase_dimord,
                                    time=waveData.DataBuckets[dataBucketName].get_time(),
                                    chanNames= waveData.DataBuckets[waveData.ActiveDataBucket].get_channel_names())
    waveData.add_data_bucket(PhaseDataBucket)

    # Log history
    if isinstance(ref, tuple):
        waveData.log_history(["Relative Phase", f"Phase_relative_to_chanPos_{ref[0]}, {ref[1]}"])
    elif isinstance(ref, int):
        waveData.log_history(["Relative Phase", f"Phase_relative_to_chanNr_{ref}"])
    else:
        waveData.log_history(["Relative Phase", "Phase_relative_to_time_series"])

def unwrap_phase(Data, dataBucketName=""):
    """
    Parameters
    ----------
    Data : WaveSpace.Utils.WaveData.WaveData
    dataBucketName : str, default=""

    Returns
    -------
    None
    """
    if dataBucketName == "":
        dataBucketName = Data.ActiveDataBucket
    else:
        Data.set_active_dataBucket(dataBucketName)
    # Retrieve the phase data from the active data bucket
    phaseData = np.angle(Data.DataBuckets[Data.ActiveDataBucket].get_data())
    currentDimord = Data.DataBuckets[Data.ActiveDataBucket].get_dimord()
    oldShape = phaseData.shape
    hasBeenReshaped, phaseData = force_dimord(phaseData, currentDimord , "trl_chan_time")

    # Initialize an array to hold the unwrapped phase data
    unwrappedPhaseofCurentData = np.zeros(phaseData.shape)

    for trl in range(phaseData.shape[0]):
        for chan in range(phaseData.shape[1]):
            unwrappedPhaseofCurentData[trl, chan, :] = np.unwrap(phaseData[trl, chan, :])

    if hasBeenReshaped:
        #reshape back to original dimord
        unwrappedPhaseofCurentData = np.reshape(unwrappedPhaseofCurentData, oldShape) 
    Unwrapped = wd.DataBucket(unwrappedPhaseofCurentData, "UnwrappedPhase", Data.DataBuckets[Data.ActiveDataBucket].get_dimord(),
                                      time=Data.DataBuckets[dataBucketName].get_time(),
                                      chanNames= Data.DataBuckets[Data.ActiveDataBucket].get_channel_names())
    Data.add_data_bucket(Unwrapped)
    Data.log_history(["UnwrappedPhase", "Phase_unwrapped from " + dataBucketName])

def bin_edges_from_centers(bin_centers):
    """Derive histogram bin edges from linearly or logarithmically spaced centers.

    Parameters
    ----------
    bin_centers : array-like
        Ordered histogram bin centers.

    Returns
    -------
    list of float
        Bin edges spanning the supplied centers.

    Notes
    -----
    Linear spacing uses arithmetic midpoints; nonuniform spacing uses
    geometric midpoints.
    """
    # Check if the centers are evenly spaced
    evenly_spaced = np.allclose(np.diff(bin_centers), bin_centers[1] - bin_centers[0])
    bin_edges = []
    if evenly_spaced:
        bin_width = bin_centers[1] - bin_centers[0]
        bin_edges.append(bin_centers[0] - bin_width/2)
        for i in range(len(bin_centers) - 1):
            bin_edge = (bin_centers[i] + bin_centers[i+1]) / 2
            bin_edges.append(bin_edge)
        bin_edges.append(bin_centers[-1] + bin_width/2)
    else:
        bin_edges.append(bin_centers[0] / np.sqrt(bin_centers[0] * bin_centers[1]))
        for i in range(len(bin_centers) - 1):
            bin_edge = np.sqrt(bin_centers[i] * bin_centers[i+1])
            bin_edges.append(bin_edge)
        bin_edges.append(bin_centers[-1] * np.sqrt(bin_centers[-2] * bin_centers[-1]))
    
    return bin_edges

def normalize_data(waveData, dimension = "", dataBucketName = " "):
    """Normalize data magnitude over selected dimensions.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to normalize.
    dimension : str or list of str, default=""
        Dimension name or names used for normalization.
    dataBucketName : str, default=" "
        Input data bucket. Defaults to the active bucket.

    Returns
    -------
    None
        Adds a ``<dataBucketName>_MagnitudeNormalized`` data bucket.
    """
    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimlist = dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    origShape = np.array(data.shape)
    if isinstance(dimension, str):
        if dimension in dimlist:
            normaxis = dimlist.index(dimension) 
            hasbeenreshaped = False
    elif isinstance(dimension, list):
        dimInd = []
        for thisdim in dimension:
            if thisdim not in dimlist:
                raise ValueError("dimension not found in dimord")
            dimInd.append(dimlist.index(thisdim))
        keepdims = np.setdiff1d(np.arange(len(origShape)),dimInd)
        #re-order dimensions to make sure the keepdims come first (only important if the selected dims are not all in the end)
        data = np.moveaxis(data, keepdims, np.arange(len(keepdims)))
        #flatten over dimension
        reshapehape = np.array(data.shape) #make sure we remember the shape at this point
        data = np.reshape(data, (*reshapehape[np.arange(len(keepdims))], np.product(reshapehape[len(keepdims):])))
        normaxis = -1
        hasbeenreshaped = True
    else:
        raise TypeError("dimension must be string or list of strings")

    #normalize by the mean of the absolute
    data = data / np.nanmean(np.abs(data), axis=normaxis, keepdims=True)

    if hasbeenreshaped:
        #reshape back to original dimord
        data = np.reshape(data, reshapehape)
        #re-order dimensions to original
        data = np.moveaxis(data, np.arange(len(keepdims)), keepdims)

    dataBucket = wd.DataBucket(data, dataBucketName + "_MagnitudeNormalized", 
                               dimord, 
                               time=waveData.DataBuckets[dataBucketName].get_time(),
                                chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
    waveData.add_data_bucket(dataBucket)
    waveData.log_history(["Normalize magnitude", "Normalized over " + '_'.join(dimension)])

def z_score_data(waveData, dimension = "", dataBucketName = " "):
    """Z-score data over selected dimensions.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to standardize.
    dimension : str or list of str, default=""
        Dimension name or names used for standardization.
    dataBucketName : str, default=" "
        Input data bucket. Defaults to active data bucket.

    Returns
    -------
    None
        Adds a ``<dataBucketName>_zScored`` data bucket.
    """
    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimlist = dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    origShape = np.array(data.shape)
    if isinstance(dimension, str):
        if dimension in dimlist:
            normaxis = dimlist.index(dimension) 
            hasbeenreshaped = False
    elif isinstance(dimension, list):
        dimInd = []
        for thisdim in dimension:
            if thisdim not in dimlist:
                raise ValueError("dimension not found in dimord")
            dimInd.append(dimlist.index(thisdim))
        keepdims = np.setdiff1d(np.arange(len(origShape)),dimInd)
        #re-order dimensions to make sure the keepdims come first (only important if the selected dims are not all in the end)
        data = np.moveaxis(data, keepdims, np.arange(len(keepdims)))
        #flatten over dimension
        reshapehape = np.array(data.shape) #make sure we remember the shape at this point
        data = np.reshape(data, (*reshapehape[np.arange(len(keepdims))], np.product(reshapehape[len(keepdims):])))
        normaxis = -1
        hasbeenreshaped = True
    else:
        raise TypeError("dimension must be string or list of strings")

    #normalize by the mean of the absolute
    data = data - np.nanmean(data, axis=normaxis, keepdims=True)
    data = data / np.nanstd(data, axis=normaxis, keepdims=True)

    if hasbeenreshaped:
        #reshape back to original dimord
        data = np.reshape(data, reshapehape)
        #re-order dimensions to original
        data = np.moveaxis(data, np.arange(len(keepdims)), keepdims)

    dataBucket = wd.DataBucket(data, dataBucketName + "_zScored", 
                               dimord, 
                               time=waveData.DataBuckets[dataBucketName].get_time(),
                            chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
    waveData.add_data_bucket(dataBucket)
    waveData.log_history(["z score", "z scored over " + '_'.join(dimension)])

def rescale_data_in_Bucket(waveData, dimension =None , range = (0,1),dataBucketName = ""):
    """Rescale data to a target range over selected dimensions.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        waveData object containing the data to rescale.
    dimension : list of str, default=["posx", "posy", "time"]
    range : tuple of float, default=(0, 1)
    dataBucketName : str, default=""

    Returns
    -------
    None
        Adds a ``<dataBucketName>_rescaled`` data bucket.
    """
    if dimension == None:
        dimension =['posx', 'posy', 'time']
    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimlist = dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    origShape = np.array(data.shape)
    dimInd = []
    for thisdim in dimension:
        if thisdim not in dimlist:
            raise ValueError("dimension not found in dimord")
        dimInd.append(dimlist.index(thisdim))
    keepdims = np.setdiff1d(np.arange(len(origShape)),dimInd)
    #re-order dimensions to make sure the keepdims come first (only important if the selected dims are not all in the end)
    data = np.moveaxis(data, keepdims, np.arange(len(keepdims)))
    #flatten over dimension
    reshapeshape = np.array(data.shape) #make sure we remember the shape at this point
    data = np.reshape(data, (*reshapeshape[np.arange(len(keepdims))], np.product(reshapeshape[len(keepdims):])))
    normaxis = -1
    hasbeenreshaped = True

    #rescale data to range
    data_min, data_max = np.nanmin(data, axis=normaxis, keepdims=True), np.nanmax(data, axis=normaxis, keepdims=True)
    data = (data - data_min) / (data_max - data_min) * (range[1] - range[0]) + range[0]

    if hasbeenreshaped:
        #reshape back to original dimord
        data = np.reshape(data, reshapeshape)
        #re-order dimensions to original
        data = np.moveaxis(data, np.arange(len(keepdims)), keepdims)

    dataBucket = wd.DataBucket(data, dataBucketName + "_rescaled", 
                               dimord, 
                               time=waveData.DataBuckets[dataBucketName].get_time(),
                               chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
    waveData.add_data_bucket(dataBucket)
    waveData.log_history(["rescale", "rescaled over " + '_'.join(dimension)])

def average_over_trials(waveData, trialInfo=None, dataBucketName=" ", type = None):
    """Average data across all trials or trial-information groups.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        waveData object containing the data to average.
    trialInfo : sequence or None, default=None
        Sequence of trial information corresponding to each trial in the data.
    dataBucketName : str, default=" "
    type : {"power", None}, default=None

    Returns
    -------
    None
        Adds one or more average data buckets to ``waveData``.
    """
    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimlist = dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()
    np.array(data.shape)

    # Find the 'trl' dimension
    if 'trl' in dimlist:
        trl_axis = dimlist.index('trl')
    else:
        raise ValueError("'trl' dimension not found in dimord")

    if type == 'power':
        data = np.abs(data)

    if trialInfo is None:
        # Average over all trials
        avg_data = np.mean(data, axis=trl_axis)
        avg_data = np.expand_dims(avg_data, axis=trl_axis)  # Add singleton dimension for 'trl' (necessary for all function that use 'assure consistency')
        dataBucket = wd.DataBucket(avg_data, dataBucketName + "_average", 
                                   waveData.DataBuckets[dataBucketName].get_dimord(), 
                                   time=waveData.DataBuckets[dataBucketName].get_time(),
                                chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
        waveData.add_data_bucket(dataBucket)
        waveData.log_history(["Average", "Averaged over all trials"])
    else:
        # Average over trials specified in trialInfo
        unique_trials = np.unique(trialInfo)
        for trial in unique_trials:
            trial_indices = [i for i, info in enumerate(waveData.get_trialInfo()) if info == trial]
            index_array = [slice(None)] * data.ndim
            index_array[trl_axis] = trial_indices
            avg_data = np.mean(data[tuple(index_array)], axis=trl_axis)
            avg_data = np.expand_dims(avg_data, axis=trl_axis)
            dataBucket = wd.DataBucket(avg_data, dataBucketName + "_average_" + trial, 
                                       waveData.DataBuckets[dataBucketName].get_dimord(), 
                                       time=waveData.DataBuckets[dataBucketName].get_time(),
                                       chanNames=waveData.DataBuckets[dataBucketName].get_channel_names() )
            waveData.add_data_bucket(dataBucket)
            waveData.log_history(["Average", "Averaged over trial: " + trial])

def generate_complex_timeseries(desired_frequency, signal_duration, sampling_frequency, pre_oscillation_time, post_oscillation_time=0):
    """Generate a complex-valued time series consisting of noise,   oscillation, and zero-valued complex signal segments.

    Parameters
    ----------
    desired_frequency : float
        Oscillation frequency in Hz.
    signal_duration : float
        Oscillation duration in seconds.
    sampling_frequency : float
        Sampling frequency in Hz.
    pre_oscillation_time : float
        Duration of leading complex noise in seconds.
    post_oscillation_time : float, default=0
        Duration of trailing zero-valued samples in seconds.

    Returns
    -------
    numpy.ndarray
        Complex time series formed by concatenating the three segments.
    """
    t_pre = np.linspace(0, pre_oscillation_time, int(pre_oscillation_time * sampling_frequency), endpoint=False)
    t_oscillation = np.linspace(0, signal_duration, int(signal_duration * sampling_frequency), endpoint=False)
    t_post = np.linspace(0, post_oscillation_time, int(post_oscillation_time * sampling_frequency), endpoint=False)

    # Create a random complex noise signal for the pre-oscillation time
    random_signal = np.random.randn(len(t_pre)) + 1j * np.random.randn(len(t_pre))

    # Create a complex oscillating signal at the desired frequency for the specified duration
    oscillation_signal = np.exp(2j * np.pi * desired_frequency * t_oscillation)

    # Concatenate the random, oscillating, and post-oscillation signals
    complex_timeseries = np.concatenate((random_signal, oscillation_signal, np.zeros_like(t_post)))

    return complex_timeseries
        
def get_freqs_from_log(waveData):
    """
    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData

    Returns
    -------
    list of int or None
    """
    # Iterate over the log history
    for log_entry in waveData.get_log_history():
        # Check if the log entry corresponds to the FFT operation
        if log_entry[0] == 'Frequency Decomposition' and log_entry[1] == 'FFT':
            # Get the frequency range string
            freq_range_str = log_entry[3]

            # Split the string to extract the frequency range and step size
            freq_range_str, freq_step_str = freq_range_str.replace('Frequency range: ', '').split(' in steps of ')

            # Split the frequency range string to extract the start and end frequencies
            freq_start, freq_end = map(int, freq_range_str.split(' to '))

            # Extract the step size
            freq_step = int(freq_step_str)

            # Generate the individual frequencies
            freqs = list(range(freq_start, freq_end, freq_step))

            return freqs

    # If no FFT log entry is found, return None
    return None

def merge_wavedata_objects(waveDataList):
    """Merge matching data buckets from multiple WaveData objects.

    Parameters
    ----------
    waveDataList : sequence of WaveSpace.Utils.WaveData.WaveData
        WaveData objects with matching channel names, bucket names, and
        dimension orders.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        Deep copy of the first object with every bucket concatenated along its
        ``trl`` dimension.

    Raises
    ------
    ValueError
        If inputs do not have matching channel names, bucket names, or
        dimension orders.
    """
    #check if all waveData objects have the same channel names, number (andd names) of dataBuckets and same dimord
    for waveData in waveDataList:
        if not (waveData.get_channel_names() == waveDataList[0].get_channel_names()):
            raise ValueError("All waveData objects must have the same channel names")
        if len(waveData.DataBuckets) != len(waveDataList[0].DataBuckets):
            raise ValueError("All waveData objects must have the same number of dataBuckets")
        # Extract the keys into a list
        keys = list(waveData.DataBuckets.keys())
        # Check if the keys are the same
        if keys != list(waveDataList[0].DataBuckets.keys()):
            raise ValueError("All waveData objects must have the same dataBuckets")
        for i in range(len(waveData.DataBuckets)):
            if waveData.DataBuckets[keys[i]].get_dimord() != waveDataList[0].DataBuckets[keys[i]].get_dimord():
                raise ValueError("All waveData objects must have the same dimord")

    # Create a new waveData object to hold the merged data
    merged_waveData = copy.deepcopy(waveDataList[0])

    # Merge dataBuckets
    for key in keys:
        # Find the index of the "trl" dimension
        dimord_list = waveDataList[0].DataBuckets[key].get_dimord().split('_')
        trl_index = dimord_list.index('trl')        
        # Concatenate the data along the "trl" dimension
        print(key)
        print([waveData.DataBuckets[key].get_data().shape for waveData in waveDataList])
        merged_data = np.concatenate([waveData.DataBuckets[key].get_data() for waveData in waveDataList], axis=trl_index)
        # Create a new DataBucket with the merged data
        merged_waveDataBucket = wd.DataBucket(
            merged_data, 
            key, 
            waveDataList[0].DataBuckets[key].get_dimord(),
            time=waveDataList[0].get_time(),
            chanNames=waveDataList[0].DataBuckets[key].get_channel_names())
        merged_waveData.add_data_bucket(merged_waveDataBucket)
    merged_waveData.log_history(["merged waveData objects", "trl in dimord corresponds to index of merged object"])


    return merged_waveData

def cosine_similarity_complex(a, b):
    """
    Parameters
    ----------
    a, b : numpy.ndarray

    Returns
    -------
    float
    """
    return np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))

def find_wave_motifs(waveData, dataBucketName=None, oscillationThresholdDataBucket = None, oscillationThresholdFlag = False, baselinePeriod = None, threshold = .7, nTimepointsEdge = 50, mergeThreshold =.6, minFrames = 10, pixelThreshold = .3, magnitudeThreshold = .1, dataInds = None, Mask = None):
    """Identify recurring, spatially similar motifs in optical-flow maps.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing complex optical-flow vectors.
    dataBucketName : str, default=None
        Name of the input flow bucket. By default, the active data bucket is
        used.
    oscillationThresholdDataBucket : str, default=None
        Name of a data bucket used for optional oscillation-power thresholding.
    oscillationThresholdFlag : bool, default=False
        Enable oscillation-power thresholding.
    baselinePeriod : tuple of int, default=None
        Start and stop sample indices defining the threshold baseline.
    threshold : float, default=0.7
        Minimum cosine similarity for consecutive frames in one motif.
    nTimepointsEdge : int, default=50
        Number of samples excluded from the beginning and end of each trial.
    mergeThreshold : float, default=0.6
        Minimum cosine similarity required to merge candidate motifs.
    minFrames : int, default=10
        Minimum consecutive-frame duration of a motif.
    pixelThreshold : float, default=0.3
        Minimum proportion of spatial positions satisfying the relevant
        threshold.
    magnitudeThreshold : float, default=0.1
        Minimum optical-flow vector magnitude.
    dataInds : tuple, default=None
        Indices selecting data to analyse. By default, all data are used.
    Mask : numpy.ndarray or bool, default=None
        Spatial mask. ``True`` uses the ``Mask`` data bucket; by default, all
        spatial positions are included.

    Returns
    -------
    list of dict
        Identified motifs. Each dictionary includes the average flow map, the
        trials and frame ranges where it occurs, and its frame count.
    """
    if dataBucketName is None:
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    
    if oscillationThresholdFlag and oscillationThresholdDataBucket is not None:        
        oscillationThresholdData = np.abs(waveData.get_data(oscillationThresholdDataBucket)[dataInds])
    else:
        oscillationThresholdData = None

    assure_consistency(waveData)
    oldShape = waveData.DataBuckets[waveData.ActiveDataBucket].get_data().shape
    currentDimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    if dataInds is None:
        dataInds = slice(None)
    data=waveData.DataBuckets[waveData.ActiveDataBucket].get_data()[dataInds]    
    splitDimord = currentDimord.split('_')	
    if 'posx' in splitDimord and 'posy' in splitDimord:
        shape = ( oldShape[splitDimord.index('posx')], oldShape[splitDimord.index('posy')])
    elif 'chan' in splitDimord:
        shape = (oldShape[splitDimord.index('chan')])

    if isinstance(Mask, np.ndarray) and not isinstance(Mask, bool):
        maskInds = np.where(Mask == 1)      
    elif Mask: 
        if 'Mask' not in waveData.DataBuckets:
            raise ValueError("No Mask dataBucket found")
        maskInds = np.where(waveData.get_data('Mask'))
        Mask = waveData.get_data('Mask')    
    else: 
        maskInds = np.where(np.ones(shape))
        Mask = np.ones(shape, dtype=bool)

    minPixels = np.sum(Mask)*pixelThreshold
    #mask and flatten data to one spatial dimension  
    data = data[:, Mask, :]

    epsilon = 1e-18  # to avoid diving by zero later
    framesequence = []
    temp_stable_patterns = []

    for trl, trial in enumerate(data):
        tStart = 0
        t = 1  
        # Calculate the baseline power
        if oscillationThresholdFlag:
            if baselinePeriod:
                baselinePower = np.mean(oscillationThresholdData[trl,  :, baselinePeriod[0]:baselinePeriod[1]], axis=(-2,-1))
                baselineStd = np.std(oscillationThresholdData[trl,  :, baselinePeriod[0]:baselinePeriod[1]], axis=(-2,-1))
                oscillationThreshold = baselinePower + baselineStd  # 2 standard deviations above the mean
            else:
                baselinePower = np.mean(oscillationThresholdData[trl,:,nTimepointsEdge:-nTimepointsEdge])  # baseline is the average power over the whole trial                
                oscillationThreshold = baselinePower *50/100  # 50% of the average power 

        while tStart < trial.shape[-1]-1:            
            templateFrame = trial[:, tStart]
            framesequence = [templateFrame]
            for t in range(tStart+1, trial.shape[-1]):
                if t < nTimepointsEdge or t > trial.shape[-1] - nTimepointsEdge:
                    if len(framesequence) >= minFrames :
                        temp_stable_patterns.append({'average': np.mean(framesequence, axis=0), 'trial': [trl], 'frames': [(tStart, t-1)], 'num_frames': t-tStart})
                    tStart = t
                    framesequence = []
                    break
                if oscillationThresholdFlag:
                    # Count the number of x,y positions that meet the power threshold
                    num_pixels_above_threshold = (oscillationThresholdData[trl,:,t] > oscillationThreshold).sum()
                    if num_pixels_above_threshold < minPixels:
                        if len(framesequence) >= minFrames :
                            temp_stable_patterns.append({'average': np.mean(framesequence, axis=0), 'trial': [trl], 'frames': [(tStart, t-1)], 'num_frames': t-tStart})
                        tStart = t
                        framesequence = []
                        break
                    powermask = oscillationThresholdData[trl,:,t] > oscillationThreshold
                testFrame = trial[:, t]                
                normalized_templateFrame = templateFrame /np.abs(templateFrame+ epsilon)
                normalized_testFrame = testFrame /np.abs(testFrame+ epsilon)
                magnitude_mask = np.abs(testFrame) > magnitudeThreshold
                if oscillationThresholdFlag:
                    magnitude_mask = np.logical_and(magnitude_mask, powermask)

                cosine_similarity = np.real(normalized_testFrame * np.conj(normalized_templateFrame))
                meets_threshold = (cosine_similarity[magnitude_mask] > threshold).sum() >= minPixels 
                               
                if meets_threshold:
                    framesequence.append(testFrame)
                else:
                    if len(framesequence) >= minFrames:
                        temp_stable_patterns.append({'average': np.mean(framesequence, axis=0), 'trial': [trl], 'frames': [(tStart, t)], 'num_frames': t-tStart+1})
                    tStart = t
                    framesequence=[]
                    break
            else:  # This is executed if the for loop completed normally
                tStart = t + 1
        if len(framesequence) >= minFrames :
            temp_stable_patterns.append({'average': np.mean(framesequence, axis=0), 'trial': [trl], 'frames': [(tStart, t)], 'num_frames': t-tStart+1})
    print(f"temporally stable sequences: {len(temp_stable_patterns)}")

    # for pattern in temp_stable_patterns:
    #     print(f"Trial: {pattern['trial']}, Frames: {pattern['frames']}")

    motifs = []     
       
    if temp_stable_patterns:
        motifs.append(temp_stable_patterns.pop(0))        

        for pattern in temp_stable_patterns:
            has_merged = False
            for motif in motifs:
                normalized_motif = motif['average'] / np.abs(motif['average'])
                normalized_pattern = pattern['average'] / np.abs(pattern['average'])
                cosine_similarity = np.real(normalized_motif * np.conj(normalized_pattern))
                if np.sum(cosine_similarity >= mergeThreshold) >= minPixels:
                    # print('Merging')
                    # a= motif['average']
                    # b= pattern['average']
                    # plt.quiver(np.real(a), np.imag(a))
                    # plt.quiver(np.real(b), np.imag(b))
                    # plt.show()
                    total_frames = motif['num_frames'] + pattern['num_frames']
                    motif['average'] = (motif['average'] * motif['num_frames'] + pattern['average'] * pattern['num_frames']) / total_frames
                    motif['num_frames'] = total_frames
                    motif['trial'].extend(pattern['trial'])
                    motif['frames'].extend(pattern['frames'])
                    has_merged = True
                    break
                # else:
                #     print('Not merging')
                #     a= motif['average']
                #     b= pattern['average']
                #     plt.quiver(np.real(a), np.imag(a))
                #     plt.quiver(np.real(b), np.imag(b))
                #     plt.show()
            if not has_merged:
                motifs.append(pattern)
                has_merged = False

        # After the loop, convert the motifs to the desired format
        for motif in motifs:
            motif['trial_frames'] = list(zip(motif['trial'], motif['frames']))
            del motif['trial']
            del motif['frames']   
            temp = np.zeros(shape, dtype=np.complex128)
            temp[maskInds] = motif['average']
            motif['average'] =  temp   
    
        print(f"Number of motifs after merging: {len(motifs)}")
    else:
        motifs = []
        print('nothing to merge')
    


    return motifs


def merge_motifs_across_subjects(motifs, mergeThreshold = .6, pixelThreshold = 1):
    """
    Merges motifs across subjects for a specific condition and frequency based on their cosine similarity.

    Parameters
    ----------
    motifs : list of dict
        Motifs with ``average``, ``subject``, ``trial_frames``, and
        ``num_frames`` entries.
    mergeThreshold : float, default=0.6
        Minimum cosine similarity for spatial positions to support a merge.
    pixelThreshold : float, default=1
        Minimum proportion of spatial positions meeting ``mergeThreshold``.

    Returns
    -------
    list of dict
        Merged motifs with subject-labelled trial-frame entries.
    """
    merged_motifs = []
    minPixels = motifs[0]['average'].shape[0]*motifs[0]['average'].shape[1]*pixelThreshold
    NotMergeCount = 0
    MergeCount = 0

    mask = motifs[0]['average'] != 0
    epsilon = 1e-18
    if not merged_motifs:
        first_motif = motifs.pop(0)
        first_motif['trial_frames'] = [(first_motif['subject'], tf) for tf in first_motif['trial_frames']]
        merged_motifs.append(first_motif)

    for motif in motifs:
        has_merged = False
        motif['trial_frames'] = [(motif['subject'], tf) for tf in motif['trial_frames']]
        for merged_motif in merged_motifs:
            normalized_merged_motif = merged_motif['average'][mask] / (np.abs(merged_motif['average'][mask]) + epsilon)
            normalized_motif = motif['average'][mask] / (np.abs(motif['average'][mask]) + epsilon)
            cosine_similarity = np.real(normalized_merged_motif * np.conj(normalized_motif))
            if np.sum(cosine_similarity >= mergeThreshold) >= minPixels:
                MergeCount += 1
                total_frames = merged_motif['num_frames'] + motif['num_frames']
                merged_motif['average'] = (merged_motif['average'] * merged_motif['num_frames'] + motif['average'] * motif['num_frames']) / total_frames
                
                # Extend 'trial_frames' list of merged_motif with 'trial_frames' list of motif
                merged_motif['trial_frames'].extend(motif['trial_frames'])

                merged_motif['num_frames'] = total_frames
                has_merged = True
                break
            else:
                NotMergeCount += 1

        if not has_merged:
            merged_motifs.append(motif)
    print ('MergeCount:', MergeCount)
    print ('NotMergeCount:', NotMergeCount)
    return merged_motifs

def match_motifs_to_templates(motifs, templates=None, mergeThreshold = None, pixelThreshold = 1):
    """Assign motifs to spatial templates using cosine similarity.

    Parameters
    ----------
    motifs : list of dict
        Motifs with ``average``, ``subject``, ``trial_frames``, and
        ``num_frames`` entries.
    templates : list of dict or numpy.ndarray or None, default=None
        Existing templates or template arrays. When None, the first motif
        average initializes one template.
    mergeThreshold : float or None, default=None
        Similarity criterion. None assigns every motif to its most similar
        template; a number requires enough matching pixels.
    pixelThreshold : float, default=1
        Minimum proportion of matching pixels when ``mergeThreshold`` is set.

    Returns
    -------
    list of dict
        Templates updated with assigned motif trial frames and frame counts.
    """
    minPixels = motifs[0]['average'].shape[0] * motifs[0]['average'].shape[1] * pixelThreshold
    epsilon = 1e-18
    # Ensure templates is a list of dicts with required fields
    if templates is None:
        templates = [{'average': motifs[0]['average'], 'num_frames': 0, 'trial_frames': []}]
    elif isinstance(templates, np.ndarray):
        templates = [{'average': templates, 'num_frames': 0, 'trial_frames': []}]
    elif isinstance(templates, list) and not isinstance(templates[0], dict):
        templates = [{'average': t, 'num_frames': 0, 'trial_frames': []} for t in templates]

    for motif in motifs:        
        motif_trial_frames = [(motif['subject'], tf) for tf in motif['trial_frames']]
        motif_num_frames = motif['num_frames']

        # Compute similarity to each template
        cosine_similarity_array = np.zeros(len(templates))
        for templateInd, template in enumerate(templates):
            normalized_template = template['average'] / (np.abs(template['average'])+ epsilon)
            normalized_motif = motif['average'] / (np.abs(motif['average']) + epsilon)
            sim = np.real(normalized_template * np.conj(normalized_motif))
            if mergeThreshold is not None:
                cosine_similarity_array[templateInd] = np.sum(sim >= mergeThreshold)
            else:
                cosine_similarity_array[templateInd] = np.sum(sim)
        print(cosine_similarity_array)
        top_template_index = np.argmax(cosine_similarity_array)
        if mergeThreshold is not None:
            if cosine_similarity_array[top_template_index] >= minPixels:
                templates[top_template_index]['trial_frames'].extend(motif_trial_frames)
                templates[top_template_index]['num_frames'] += motif_num_frames
        else: #winner takes all, whatever the actual similarity
            templates[top_template_index]['trial_frames'].extend(motif_trial_frames)
            templates[top_template_index]['num_frames'] += motif_num_frames

    # The 'average' field for each template remains unchanged (the template itself)
    return templates




def nan_gradient(data, dx, dy, dz):
    """Calculate centered phase gradients while preserving NaN boundaries.

    Parameters
    ----------
    data : numpy.ndarray
        Three-dimensional phase array in radians.
    dx, dy, dz : float
        Sample spacing along the first, second, and third axes.

    Returns
    -------
    grad_x, grad_y, grad_z : numpy.ndarray
        Centered phase gradients with the same shape as ``data``. Boundary
        values are padded with NaNs.

    Notes
    -----
    Wrapped phase differences are calculated through complex unit vectors
    before centered averaging.
    credit: https://stackoverflow.com/questions/71585733/
    """
    # Compute differences along each axis and unwrap the phase differences
    diff_x = np.angle(np.exp(1j * (data[1:, ...] - data[:-1, ...]))) / dx
    diff_y = np.angle(np.exp(1j * (data[:, 1:, :] - data[:, :-1, :]))) / dy
    diff_z = np.angle(np.exp(1j * (data[..., 1:] - data[..., :-1]))) / dz

    # Average gradients to center them in the grid
    grad_x = np.nanmean([diff_x[1:], diff_x[:-1]], axis=0)
    grad_y = np.nanmean([diff_y[:, 1:, :], diff_y[:, :-1, :]], axis=0)
    grad_z = np.nanmean([diff_z[..., 1:], diff_z[..., :-1]], axis=0)

    # Pad to maintain shape consistency
    grad_x = np.pad(grad_x, ((1, 1), (0, 0), (0, 0)), constant_values=np.nan)
    grad_y = np.pad(grad_y, ((0, 0), (1, 1), (0, 0)), constant_values=np.nan)
    grad_z = np.pad(grad_z, ((0, 0), (0, 0), (1, 1)), constant_values=np.nan)

    return grad_x, grad_y, grad_z