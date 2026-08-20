#%%
import matplotlib.pyplot as plt
import numpy as np
from numpy import matlib

from WaveSpace.Utils import HelperFuns as hf
from WaveSpace.Utils import WaveData as wd


def assertCorrectWaveSettings(Type, ntrials, waveSettings):
    """Validate type-specific wave simulation settings.

    Parameters
    ----------
    Type : str
        Simulation type passed to :func:`simulate_signal`.
    ntrials : int
        Number of simulated trials.
    waveSettings : dict
        Type-specific simulation settings.

    Returns
    -------
    None
        Raises an assertion error when required settings are absent or an
        array-like setting does not have one value per trial.

    Raises
    ------
    AssertionError
        If required settings are missing, nonlinear plane-wave settings are
        incomplete, or per-trial setting lengths do not match ``ntrials``.
    """
    # Check if all required variables are set for the type of wave
    if Type == "PlaneWave" or Type == "TargetWave" :
        assert "TemporalFrequency" in waveSettings, "TemporalFrequency not set"
        assert "SpatialFrequency" in waveSettings, "SpatialFrequency not set"
        assert "WaveDirection" in waveSettings, "WaveDirection not set" 
    if Type == "RotatingWave": 
       assert "TemporalFrequency" in waveSettings, "TemporalFrequency not set"
       assert "WaveDirection" in waveSettings, "WaveDirection not set" 
    if Type == "LocalOscillation":
        assert "TemporalFrequency" in waveSettings, "TemporalFrequency not set"
        assert "OscillatoryPhase" in waveSettings, "OscillatoryPhase not set" 
    # Check if all required variables are set
    if "NonLinearSkew" in waveSettings:
        assert "NonLinearDegree" in waveSettings and (Type == "PlaneWave") , "NonLinearSkew set without NonLinearDegree or to wrong type of data"
    if "NonLinearDegree" in waveSettings:
        assert "NonLinearSkew" in waveSettings and (Type == "PlaneWave") , "NonLinearDegree set without NonLinearSkew or to wrong type of data"
       
    # If array or list is supplied, size must be the same as amount of trials
    for item in waveSettings.items():
        if np.ma.isarray(item[1]):
            assert len(item[1])==ntrials, f"Length of supplied array for \"{item[0]}\" must equal amount of trials"

#%%
def simulate_signal(Type, ntrials, MatrixSize, SampleRate, SimDuration, SimLayout= "channels",time=None, **waveSettings):
    """Generate simulated wave or noise data as a WaveData object.

    Parameters
    ----------
    Type : str
        Simulation type. Supported values include ``"PlaneWave"``,
        ``"TargetWave"``, ``"RotatingWave"``, ``"LocalOscillation"``,
        ``"SpatialPinkNoise"``, ``"WhiteNoise"``, ``"StationaryPulse"``,
        ``"FrequencyGradient"``.
    ntrials : int
        Number of trials to simulate.
    MatrixSize : int
        Width and height of the square simulated sensor grid.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Duration of each trial in seconds.
    SimLayout : {"channels", "grid"}, default="channels"
        Spatial representation of the returned data. ``"grid"`` preserves the
        two-dimensional spatial layout; ``"channels"`` uses flattened channels.
    time : array-like, default=[]
        Explicit time values for the generated data. By default, time is
        derived from ``SampleRate`` and ``SimDuration``.
    **waveSettings
        Type-specific simulation settings. Required settings are validated for
        plane, target, rotating, and local-oscillation waves. Scalar values
        apply to all trials; two-item tuples are sampled uniformly per trial;
        arrays and lists must contain one value per trial.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        Simulated data in a ``SimulatedData`` bucket, with sensor positions and
        simulation settings attached. A ``Mask`` bucket is added when onset,
        duration, or oscillator-proportion settings are supplied.
    """
    if time is None:
        time = []
    assertCorrectWaveSettings(Type, ntrials, waveSettings)
    #InitializeDataCubes
    fullData = np.zeros((ntrials,MatrixSize,MatrixSize,int(np.floor(SampleRate*SimDuration))))

    simOptions = []
    # Create SimOptions for each Trial
    for trialNr in range(ntrials):
        currentOptions = {}
        for key, values in waveSettings.items():
            if type(values) is tuple:                    
                currentOptions[key] = np.random.uniform(values[0], values[1])
            elif isinstance(values, np.ndarray) or type(values) is list: 
                currentOptions[key] = values[trialNr]
            else:
                currentOptions[key] = values
        simOptions.append(currentOptions)
    isMaskPresent = "WaveOnset" in waveSettings or "WaveDuration" in waveSettings or "OscillatorProportion" in waveSettings
    if isMaskPresent:
        fullMask = np.zeros((ntrials,MatrixSize,MatrixSize,int(np.floor(SampleRate*SimDuration))))
    
    for TrialNr, SimOption in enumerate(simOptions):
        #Add Cube
        if Type == "None":
            fullData[TrialNr,:,:,:] = initialize_data(MatrixSize, SampleRate, SimDuration)
        if Type == "PlaneWave":
            fullData[TrialNr,:,:,:] = create_plane_wave( MatrixSize, SampleRate, SimDuration,SimOption)
            if isMaskPresent:                               
                fullMask[TrialNr,:,:,:] = create_plane_wave_mask(MatrixSize, SampleRate, SimDuration,SimOption)                
        if Type == "TargetWave":
            fullData[TrialNr,:,:,:] = create_target_wave( MatrixSize, SampleRate, SimDuration,SimOption)
        if Type == "RotatingWave":
            fullData[TrialNr,:,:,:] = create_rotating_wave( MatrixSize, SampleRate, SimDuration,SimOption)
        if Type == "LocalOscillation":
            fullData[TrialNr,:,:,:] = create_local_oscillators( MatrixSize, SampleRate, SimDuration,SimOption)
            if isMaskPresent:
                fullMask[TrialNr,:,:,:] = CreateOscillatorMask( MatrixSize, SampleRate, SimDuration, SimOption)
        if Type == "SpatialPinkNoise":
            fullData[TrialNr,:,:,:] = create_pink_noise( MatrixSize, SampleRate, SimDuration)
        if Type == "WhiteNoise":
            fullData[TrialNr,:,:,:] = create_white_noise( MatrixSize, SampleRate, SimDuration)
        if Type == "StationaryPulse":
            fullData[TrialNr,:,:,:] = create_stationary_pulse(MatrixSize, SampleRate, SimDuration, SimOption)
        if Type == "FrequencyGradient":
            fullData[TrialNr,:,:,:] = create_frequency_gradient(MatrixSize, SampleRate, SimDuration, SimOption)
       
    waveData = create_wavedata(fullData, SampleRate, SimDuration, SimLayout, simOptions, time=time)  
    if isMaskPresent: 
        if (len(fullMask.shape)==4 and SimLayout != "grid"):
            fullMask = np.reshape(fullMask,(fullMask.shape[0],fullMask.shape[1]*fullMask.shape[2],fullMask.shape[3]), order='C') 
        if len(time) > 0:
           dataBucket = wd.DataBucket(fullMask,"Mask", waveData.DataBuckets["SimulatedData"].get_dimord(), time=time ,chanNames= waveData.get_channel_names())   
        else:
            dataBucket = wd.DataBucket(fullMask,"Mask", waveData.DataBuckets["SimulatedData"].get_dimord(), sampleRate=waveData.get_sample_rate() ,chanNames= waveData.get_channel_names())   
        waveData.add_data_bucket(dataBucket)
    return waveData

def initialize_data(MatrixSize, SampleRate,SimDuration):
    """Create an empty square spatiotemporal simulation array.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.

    Returns
    -------
    numpy.ndarray
        Zero-filled array ordered as x position, y position, and time.
    """
    return np.zeros((MatrixSize,MatrixSize,int(np.floor(SimDuration * SampleRate))))  

def create_wavedata(data, SampleRate, SimDuration, SimLayout, simOptions, name = "SimulatedData", time=None):
    """Create a WaveData object from simulated array data.

    Parameters
    ----------
    data : numpy.ndarray
        Simulated data ordered as trials, x position, y position, and time or
        as trials, channels, and time.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Duration of each simulation trial in seconds.
    SimLayout : {"channels", "grid"}
        Output spatial layout. ``"grid"`` reshapes square channels into
        ``posx_posy`` dimensions.
    simOptions : list of dict
        Per-trial simulation settings stored as simulation metadata.
    name : str, default="SimulatedData"
        Name of the output data bucket.
    time : array-like, default=[]
        Explicit time vector. When omitted, it is derived from the sample rate.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the simulated bucket, regular-grid channel
        metadata, and simulation settings.
    """
    #flatten channels
    if time is None:
        time = []
    if (len(data.shape)==4):
       data = np.reshape(data,(data.shape[0],data.shape[1]*data.shape[2],data.shape[3]), order='C') 
    dimord = "trl_chan_time"

    waveData = wd.WaveData(sampleRate=SampleRate, time=(0,SimDuration))
    x_ = np.linspace(0, int(np.sqrt(data.shape[1]-1)), int(np.sqrt(data.shape[1])))
    y_ = np.linspace(0, int(np.sqrt(data.shape[1]-1)), int(np.sqrt(data.shape[1])))
    grid = np.meshgrid(x_, y_ , indexing='xy')
    waveData.HasRegularLayout = True
    chanpos = np.ones([3,data.shape[1]]).T
    chanpos[:,0:2] = (np.vstack(list(map(np.ravel, grid)))).T  
    waveData.log_history("Created SimulatedData")
    waveData.set_simInfo(simOptions)
    waveData.set_channel_positions(chanpos)
    waveData.set_channel_names([str(s) for s in np.arange(len(chanpos))])
    if len(time) > 0: 
        dataBucket = wd.DataBucket(data,name, dimord, time=time ,chanNames= waveData.get_channel_names(), unit="AU")
    else:
        dataBucket = wd.DataBucket(data,name, dimord, sampleRate=waveData.get_sample_rate() ,chanNames= waveData.get_channel_names(), unit="AU")
    waveData.add_data_bucket(dataBucket)

    if SimLayout == "grid":
        hf.squareSpatialPositions(waveData)    
    return waveData

def apply_mask(signal, mask):
    """Suppress signal values where a mask is 1.

    Parameters
    ----------
    signal : numpy.ndarray
        Signal array.
    mask : numpy.ndarray
        Mask broadcastable to ``signal`` where 1 denotes suppression.

    Returns
    -------
    numpy.ndarray
        Masked signal calculated as ``signal * (1 - mask)``.
    """
    return signal * (1-mask)

def create_stationary_pulse(MatrixSize, SampleRate, SimDuration, SimOptions):
    """Generate a spatially stationary Gaussian pulse with temporal oscillation.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOptions : dict
        Settings containing ``CenterX``, ``CenterY``, ``Sigma``, and
        ``TemporalFrequency``.

    Returns
    -------
    numpy.ndarray
        Array ordered as x position, y position, and time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    _, _, npoints = signalCube.shape    
    grid = get_board(MatrixSize)
    centerX = SimOptions["CenterX"]
    centerY = SimOptions["CenterY"]
    test = gaussian2d(x=grid[0], y=grid[1], x0=centerX, y0=centerY, sigma=SimOptions["Sigma"])     
    signalCube = np.repeat(test[:,:,np.newaxis], npoints, axis=2)
    time_vect = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    signalCubeOut = signalCube * np.sin(2 * np.pi * SimOptions["TemporalFrequency"] * time_vect)
    return signalCubeOut 

def gaussian2d(x, y, x0, y0, sigma):
    """Evaluate a two-dimensional isotropic Gaussian.at (x, y) with the given center point (x0, y0) and standard deviation sigma.

    Parameters
    ----------
    x, y : numpy.ndarray or float
        Evaluation coordinates.
    x0, y0 : float
        Gaussian center coordinates.
    sigma : float
        Gaussian standard deviation.

    Returns
    -------
    numpy.ndarray or float
        Gaussian values at the supplied coordinates.
    """
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

def create_plane_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate a travelling plane wave across a square spatial grid.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``TemporalFrequency``, ``SpatialFrequency``, and
        ``WaveDirection``. Optional ``NonLinearDegree`` and ``NonLinearSkew``
        control waveform nonlinearity.

    Returns
    -------
    numpy.ndarray
        Plane-wave values ordered as x position, y position, and time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration) 
    NonLinearDegree = 0
    NonLinearSkew = 0
    if ("NonLinearDegree" in SimOption):
        NonLinearDegree = SimOption["NonLinearDegree"]
    if ("NonLinearSkew" in SimOption):
        NonLinearSkew = SimOption["NonLinearSkew"]

    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    orientation = np.deg2rad(360-SimOption["WaveDirection"])       

    #Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
        [np.sin(orientation), np.cos(orientation)]]
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[ii,jj] , Y[ii,jj]]).T)                
            M[ii,jj] = tmp_M[0] 
    
    L = 1/SimOption["SpatialFrequency"] * MatrixSize # from cycles per image to shift per gridstep

    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Adapted from:
            #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py       
            
            time_vect = np.linspace(0, SimDuration , int(SimDuration  * SampleRate ))

            factor = np.sqrt(1 - NonLinearDegree**2)

            num = NonLinearDegree * np.sin(NonLinearSkew) / (1 + factor)            
            num = num + np.sin(2 * np.pi * SimOption["TemporalFrequency"]* time_vect - 2*np.pi/L*M[ii,jj])

            denom = 1 - NonLinearDegree * np.cos(2 * np.pi * SimOption["TemporalFrequency"] \
                    * time_vect + NonLinearSkew - 2*np.pi/L*M[ii,jj])

            signalCube[ii,jj,:] = factor * (num / denom)
            #L*M[ii,jj] is the phaseshift over space where M is a linear 
            # gradient over the grid in the direction determined by orientation
            # And L is the stepsize (determined by spatial frequency)
    return signalCube

def create_plane_wave_mask(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate a spatiotemporal onset and offset mask for a plane wave.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing wave direction, temporal and spatial frequencies,
        ``WaveOnset``, and ``WaveDuration``.

    Returns
    -------
    numpy.ndarray
        Mask ordered as x position, y position, and time. One denotes masked
        samples and zero denotes active wave samples.
    """
    MaskCube = initialize_data(MatrixSize, SampleRate, SimDuration)  
    orientation = np.deg2rad(360-SimOption["WaveDirection"])      
    grid = get_board(MatrixSize)          
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    #Create Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
            [np.sin(orientation), np.cos(orientation)]]
    for timeSample in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[timeSample,jj] , Y[timeSample,jj]]).T)                
            M[timeSample,jj] = tmp_M[0] 
    
    #M = scale(M,(0,1))
    # Create Time-vector
    temporalChange = SimOption["TemporalFrequency"] / SampleRate         
    spatialChange = SimOption["SpatialFrequency"] /MatrixSize
    stepsize = temporalChange * (1 / spatialChange)
    onsetTimeVector = np.arange(np.min(M),np.max(M)+stepsize,stepsize)            
    #time = time
    onsetStartingIndex = int(np.floor(SimOption["WaveOnset"] / (1000/SampleRate)))
    offsetStartingIndex = len(onsetTimeVector) +  onsetStartingIndex +  int(np.floor(SimOption["WaveDuration"]/ (1000/SampleRate)))

    for timeSample in range(int(np.floor(SimDuration * SampleRate))):
        #Mask ON
        if (timeSample < onsetStartingIndex):
            MaskCube[:,:,timeSample]  = 1.0

        #Signal Onset        
        if (timeSample < (len(onsetTimeVector) + onsetStartingIndex)) and (timeSample >= onsetStartingIndex) :            
            onset = onsetTimeVector[timeSample - onsetStartingIndex]
            MatrixFunction = np.vectorize(lambda a, onset=onset: 0.0 if a <= onset else 1.0)
            MaskCube[:,:,timeSample]  = MatrixFunction(M)

        # no mask Sustain
        if (timeSample >= len(onsetTimeVector) + onsetStartingIndex) and (timeSample < offsetStartingIndex):
            MaskCube[:,:,timeSample]  = 0.0

        #Offset
        if (timeSample >= offsetStartingIndex) and (timeSample < offsetStartingIndex + len(onsetTimeVector)):            
            offset = onsetTimeVector[timeSample - offsetStartingIndex]
            MatrixFunction = np.vectorize(lambda a, offset=offset: 1.0 if a <= offset else 0.0)
            MaskCube[:,:,timeSample]  = MatrixFunction(M)
        #Signal Off
        if (timeSample >= offsetStartingIndex + len(onsetTimeVector)):
            MaskCube[:,:,timeSample]  = 1.0
    return MaskCube

def create_frequency_gradient(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate oscillations with a frequency gradient along a spatial dimension.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``WaveDirection``, ``MinTemporalFrequency``, and
        ``MaxTemporalFrequency``. Optional nonlinear settings are supported.

    Returns
    -------
    numpy.ndarray
        Frequency-gradient signal ordered as x position, y position, and time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration) 
    NonLinearDegree = 0
    NonLinearSkew = 0
    if ("NonLinearDegree" in SimOption):
        NonLinearDegree = SimOption["NonLinearDegree"]
    if ("NonLinearSkew" in SimOption):
        NonLinearSkew = SimOption["NonLinearSkew"]

    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1]
    M= np.zeros(grid[0].shape)
    orientation = np.deg2rad(360-SimOption["WaveDirection"])       

    #Rotation matrix
    R = [[np.cos(orientation),- np.sin(orientation) ],\
        [np.sin(orientation), np.cos(orientation)]]
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Generate gradient in direction of orientation
            tmp_M = np.matmul(R,np.array([X[ii,jj] , Y[ii,jj]]).T)                
            M[ii,jj] = tmp_M[0] 
    FrequencyGradient = hf.scale(M, (SimOption["MinTemporalFrequency"], SimOption["MaxTemporalFrequency"]))

    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Adapted from:
            #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py       
            
            time_vect = np.linspace(0, SimDuration , int(SimDuration  * SampleRate ))

            factor = np.sqrt(1 - NonLinearDegree**2)

            num = NonLinearDegree * np.sin(NonLinearSkew) / (1 + factor)            
            num = num + np.sin(2 * np.pi * FrequencyGradient[ii,jj]* time_vect)

            denom = 1 - NonLinearDegree * np.cos(2 * np.pi * FrequencyGradient[ii,jj] \
                    * time_vect + NonLinearSkew)

            signalCube[ii,jj,:] = factor * (num / denom)

    return signalCube

def create_target_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate a radial target wave around a specified spatial center.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``CenterX``, ``CenterY``, ``TemporalFrequency``,
        ``SpatialFrequency``, and signed ``WaveDirection``.

    Returns
    -------
    numpy.ndarray
        Real-valued target-wave signal ordered as x position, y position, and
        time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)  
    grid = get_board(MatrixSize)    
    X = grid[0]
    Y = grid[1] 
    D = np.sqrt((X-SimOption["CenterX"]) **2  + (Y-SimOption["CenterY"])**2)

    # wavelength
    L = 1/SimOption["SpatialFrequency"] * MatrixSize

    # direction of wave
    freq_sign = np.sign(SimOption["WaveDirection"])
    #Time vector
    time_vect = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    #Loop through positions
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            #Radial wave
            signalCube[ii,jj,:] = np.real(np.exp(freq_sign * 1j * ( 2 * np.pi * np.abs(SimOption["TemporalFrequency"]) * \
                time_vect -2 * np.pi / L * D[ii,jj])))
    if SimOption["WaveDirection"] < 0:
        return np.flip(signalCube, axis=2)
    else:
        return signalCube

def create_rotating_wave(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate a rotating wave.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``TemporalFrequency`` and signed ``WaveDirection``.

    Returns
    -------
    numpy.ndarray
        Complex rotating-wave signal ordered as x position, y position, and
        time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    grid = get_board(MatrixSize)    
    X = grid[0] 
    Y = grid[1] 

    [_R,TH] = cart2pol(X,Y)
    # direction of wave
    freq_sign = np.sign(SimOption["TemporalFrequency"])
    direction = np.sign(SimOption["WaveDirection"] )
    #Time vector
    time_vect = np.linspace(0, SimDuration , int(SimDuration *SampleRate ))
    for ii in range(MatrixSize):
        for jj in range(MatrixSize):
            signalCube[ii,jj,:] = np.real(np.exp(freq_sign * 1j * (2*np.pi*np.abs(SimOption["TemporalFrequency"]) * 
                time_vect - direction * TH[ii,jj])))
    return signalCube

def create_local_oscillators(MatrixSize, SampleRate, SimDuration,SimOption):
    """Generate synchronized or random-phase local oscillators.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``TemporalFrequency`` and ``OscillatoryPhase`` as
        either ``"Random"`` or ``"Synchronized"``.

    Returns
    -------
    numpy.ndarray
        Oscillator values ordered as x position, y position, and time.
    """
    signalCube = initialize_data(MatrixSize, SampleRate, SimDuration)
    time = np.linspace(0,SimDuration , int( SimDuration * SampleRate ))
    if SimOption["OscillatoryPhase"] == "Random": 
        for ii in range(MatrixSize):
            for jj in range(MatrixSize):
                #adds sine to initial value in fullstatus 
                signal = np.sin(2*np.pi*SimOption["TemporalFrequency"]* time + \
                    np.random.choice(np.arange(0,2*np.pi),1))
                signalCube[ii,jj,:] = signal
    if SimOption["OscillatoryPhase"] == "Synchronized": 
        for ii in range(MatrixSize):
            for jj in range(MatrixSize):
                #adds sine to initial value in fullstatus 
                signal = np.sin(2*np.pi*SimOption["TemporalFrequency"]* time) #add + Phase offset
                signalCube[ii,jj,:] = signal
    return signalCube

def CreateOscillatorMask(MatrixSize, SampleRate, SimDuration, SimOption):
    """Generate a random mask for a proportion of local oscillators.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.
    SimOption : dict
        Settings containing ``OscillatorProportion``.

    Returns
    -------
    numpy.ndarray
        Mask ordered as x position, y position, and time. 1 marks selected
        spatial positions.
    """
    # selects which cells will be oscillating
    proportionOfOscillators = SimOption["OscillatorProportion"]
    oscillatorIndeces = np.random.choice(MatrixSize * MatrixSize, int(np.floor((MatrixSize**2) * (1-proportionOfOscillators))),replace=False)
    oscillatorIndeces = np.unravel_index(oscillatorIndeces, (MatrixSize, MatrixSize))  
    oscillatorMask = np.zeros((MatrixSize, MatrixSize, int(SimDuration * SampleRate)))
    oscillatorMask[oscillatorIndeces[0], oscillatorIndeces[1], :] = 1
    return oscillatorMask

def create_white_noise( MatrixSize, SampleRate, SimDuration):
    """Generate spatially and temporally independent Gaussian noise.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.

    Returns
    -------
    numpy.ndarray
        White-noise array ordered as x position, y position, and time.
    """
    return np.random.randn(MatrixSize,MatrixSize, int(np.floor(SimDuration*SampleRate)))

def create_pink_noise( MatrixSize, SampleRate, SimDuration):
    """Generate spatially filtered pink-noise-like data.

    Parameters
    ----------
    MatrixSize : int
        Number of positions along each spatial dimension.
    SampleRate : float
        Sampling frequency in Hz.
    SimDuration : float
        Simulation duration in seconds.

    Returns
    -------
    numpy.ndarray
        Rescaled noise array ordered as x position, y position, and time.

    Notes
    -----
    Each time point is filtered in the two-dimensional Fourier domain using a
    spatial spectrum with exponent two.
    """
    signalCube = np.random.randn(MatrixSize,MatrixSize, int(np.floor(SimDuration * SampleRate))) 

    for i in range(int(np.floor(SimDuration * SampleRate))):        
        beta = 2
        u = np.concatenate((np.arange(0,(int(np.floor(MatrixSize)/2)+1),1), np.arange(-(int(np.floor(MatrixSize)/2)-1),0,1)))/MatrixSize
        u = matlib.repmat(u,MatrixSize, 1)
        v = u.T
        SF = (u**2 + v**2)**(beta / 2)
        SF[np.inf == SF] = 0
        # phi=(np.reshape(np.arange(0,1,1/(16*16)),[16,16])).T
        #Take timepoint over space
        phi=scale(signalCube[:,:,i],[0,2*np.pi]).T
        FFT_signal = (SF**.5 *(np.cos(2*np.pi*phi)+1j*np.sin(2*np.pi*phi))).T
        FFT_signal = np.fft.fftshift(FFT_signal)
        FFT_signal[0,0] = 0
        FFT_signal = (FFT_signal * np.exp(1j*(phi)))
        FFT_signal = np.real(np.fft.ifft2(FFT_signal))
        status = scale(FFT_signal)
        signalCube[:,:,i] = status
    return signalCube

def SNRMix(SignalWaveData, NoiseWaveData, SNR, Mask=None, SimLayout="channels"):
    """Mix simulated signal and noise data at a specified signal-to-noise ratio.

    Parameters
    ----------
    SignalWaveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the simulated signal. Its first data bucket
        is used as the signal input.
    NoiseWaveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing noise in its active data bucket.
    SNR : float, int, or numpy.ndarray
        Signal weighting. A scalar applies to all samples; arrays may match the
        signal shape or provide one value per trial.
    Mask : numpy.ndarray, default=None
        Mask that suppresses the signal contribution where its value is 1.
        When omitted, the ``Mask`` bucket in ``SignalWaveData`` is used when
        available.
    SimLayout : str, default="channels"
        Spatial layout passed to the returned simulated WaveData object.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        New WaveData object containing ``(noise + signal * SNR) / (1 + SNR)``.
    """
    if Mask is not None and np.any(Mask):
        SNR = SNR * (1-Mask)
    elif "Mask" in SignalWaveData.DataBuckets:
        SNR = SNR * (1-SignalWaveData.DataBuckets["Mask"].get_data())
    
    signal = SignalWaveData.DataBuckets[next(iter(SignalWaveData.DataBuckets.keys()))].get_data()
    noise = NoiseWaveData.get_active_data()
    if isinstance(SNR, (float, int)) or SNR.shape == signal.shape:
        signalCube = (noise + (signal * SNR)) / (1 + SNR)
    else:
        signalCube = (noise + (signal * SNR[:,np.newaxis,np.newaxis])) / (1 + SNR)[:,np.newaxis,np.newaxis]
    
    wavedata = create_wavedata(signalCube, SignalWaveData.get_sample_rate(), SignalWaveData.get_time()[-1],SimLayout, SignalWaveData.get_SimInfo())
    return wavedata

# utility functions

def createVectorField(board):
    """Create a normalized two-dimensional vector field from grid coordinates.

    Parameters
    ----------
    board : sequence of numpy.ndarray
        X and Y coordinate arrays, for example from :func:`get_board`.

    Returns
    -------
    x, y, u, v : numpy.ndarray
        Input coordinates and vector-field components.
    """
    # outputs to matplotlib Quiver
    x = board[0]
    y = board[1]
    u = -y/np.sqrt(x**2 + y**2)
    v = -x/np.sqrt(x**2 + y**2)
    return x,y,u,v

def getProbeColor(index, totalProbes):
    """Select an HSV color for a probe index.

    Parameters
    ----------
    index : int
        Zero-based probe index.
    totalProbes : int
        Number of probe colors to distribute across the colormap.

    Returns
    -------
    tuple of float
        RGBA color value.
    """
    cmap = plt.cm.hsv
    return cmap(index/totalProbes) 

def cart2pol(x, y):
    """Convert Cartesian coordinates to polar coordinates.

    Parameters
    ----------
    x, y : numpy.ndarray or float
        Cartesian coordinate components.

    Returns
    -------
    rho : numpy.ndarray or float
        Radial distance.
    phi : numpy.ndarray or float
        Polar angle in radians.
    """
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    """Convert polar coordinates to Cartesian coordinates.

    Parameters
    ----------
    rho : numpy.ndarray or float
        Radial distance.
    phi : numpy.ndarray or float
        Polar angle in radians.

    Returns
    -------
    x, y : numpy.ndarray or float
        Cartesian coordinate components.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_updated_colors(cmap, status):    
    """Map normalized simulation values to colormap face colors.

    Parameters
    ----------
    cmap : matplotlib.colors.Colormap
        Colormap used to transform values to colors.
    status : numpy.ndarray
        Simulation values expected to lie between -1 and 1.

    Returns
    -------
    list
        Colormap values for every flattened status element.
    """
    #fp = open('cmap.pkl', 'rb')
    #cmap = pickle.load(fp)
    #fp.close()
    #cmap = AllOptions["ColorMap"]
    status= status.astype(float)
    #values are between -1 and 1 
    # Don't use "scale()" as this will account for max and min values and change
    # colorscales between frames
    status = (status +1)/2
    facecolors = [cmap(value) for value in status.flatten()]
    return facecolors

def scale(x, out_range=(-1, 1), axis=None):
    """Linearly rescale values to a target range.

    Parameters
    ----------
    x : numpy.ndarray
        Input values.
    out_range : tuple of float, default=(-1, 1)
        Lower and upper values of the output range.
    axis : int or None, default=None
        Axis used to calculate input minimum and maximum values.

    Returns
    -------
    numpy.ndarray
        Rescaled values.
    """
    domain = np.min(x, axis), np.max(x, axis)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def get_board(size):
    """Create centered two-dimensional coordinate arrays.

    Parameters
    ----------
    size : int
        Number of positions along each spatial dimension.

    Returns
    -------
    list of numpy.ndarray
        X and Y coordinate arrays forming a square grid.
    """
    #make a grid go from  -7 to 8 in two dimensions
    #one matrix x and one matrix y
    xs = np.arange(0, size) - (np.floor(size /2)-1)
    ys = np.arange(0, size) - (np.floor(size /2)-1)
    board = np.meshgrid(xs, ys)
    return board

def abreu2010(f, nonlin_deg, nonlin_phi, sample_rate, seconds):
    #Adapted from:
    #https://gitlab.com/emd-dev/emd/-/blob/master/emd/simulate.py
    r"""Simulate a non-linear waveform using equation 7 in [1]_.

    Parameters
    ----------
    f : float
        Fundamental frequency of generated signal
    nonlin_deg : float
        Degree of non-linearity in generated signal
    nonlin_phi : float
        Skew in non-linearity of generated signal
    sample_rate : float
        The sampling frequency of the generated signal
    seconds : float
        The number of seconds of data to generate

    References
    ----------
    [1] Abreu, T., Silva, P. A., Sancho, F., & Temperville, A. (2010).
       Analytical approximate wave form for asymmetric waves. Coastal Engineering,
       57(7), 656-667. https://doi.org/10.1016/j.coastaleng.2010.02.005
    [2] Drake, T. G., & Calantoni, J. (2001). Discrete particle model for
       sheet flow sediment transport in the nearshore. In Journal of Geophysical
       Research: Oceans (Vol. 106, Issue C9, pp. 19859-19868). American
       Geophysical Union (AGU). https://doi.org/10.1029/2000jc000611

    """
    time_vect = np.linspace(0, seconds, int(seconds * sample_rate))

    factor = np.sqrt(1 - nonlin_deg**2)
    num = nonlin_deg * np.sin(nonlin_phi) / (1 + factor)
    num = num + np.sin(2 * np.pi * f * time_vect)

    denom = 1 - nonlin_deg * np.cos(2 * np.pi * f * time_vect + nonlin_phi)

    return factor * (num / denom)

def combine_SimData(SimDataList, dimension = 'trl', SimCondList = None, dataBucketNames = None):
    """Combine compatible simulated WaveData objects by trial or time.

    Parameters
    ----------
    SimDataList : sequence of WaveSpace.Utils.WaveData.WaveData
        At least two simulated WaveData objects to combine.
    dimension : {"trl", "time"}, default="trl"
        Concatenation dimension. Trial concatenation requires matching time
        vectors; time concatenation requires matching trial counts.
    SimCondList : sequence of str, default=None
        Condition labels assigned to the combined simulation metadata. By
        default, sequential ``"Condition_<index>"`` labels are used.
    dataBucketNames : sequence of str, default=None
        Data buckets to combine. By default, all buckets in the first object
        are combined.

    Returns
    -------
    WaveSpace.Utils.WaveData.WaveData
        New WaveData object containing combined buckets, metadata, channel
        information, and trial labels.
    """
    # Check if there are at least two datasets
    assert len(SimDataList) >= 2, "At least two datasets are required"
    sampleRate = SimDataList[0].get_sample_rate()
    time = SimDataList[0].get_time()
    channel_names = SimDataList[0].get_channel_names()
    channel_positions = SimDataList[0].get_channel_positions()
    dimord = SimDataList[0].DataBuckets[SimDataList[0].ActiveDataBucket].get_dimord()
    if dataBucketNames is None:
        dataBucketNames = SimDataList[0].DataBuckets.keys()

    newdata = [None] * len(dataBucketNames)
    # If SimCondList is not provided, use the index in SimDataList as strings
    if SimCondList is None:
        SimCondList = ['Condition_' + str(i) for i in range(len(SimDataList))]
    #check that all datasets have the same dimensionality
    for SimData in SimDataList:
        for name in dataBucketNames:
            assert SimData.DataBuckets[name].get_dimord() == dimord, "Dimension order must be the same for all datasets"
    if dimension == 'trl':
        # Get the sample rate, time, channel names, and channel positions of the first dataset
        # Check if all datasets have the same sample rate, time, channel names, and channel positions
        for SimData in SimDataList[1:]:
            assert SimData.get_sample_rate() == sampleRate, "Sample Rates are not the same"
            assert np.array_equal(SimData.get_time(), time), "Time is not the same"
            assert SimData.get_channel_names() == channel_names, "Channel names are not the same"
            assert np.array_equal(SimData.get_channel_positions(), channel_positions), "Channel positions are not the same"
        for ind,name in enumerate(dataBucketNames):
            newdata[ind] = np.concatenate([SimData.get_data(name) for SimData in SimDataList], axis=0)
        SimInfo = []
        if len(SimCondList) == len(newdata[0]):
            titlecounter = 0
            for simdata in SimDataList:
                sim_info = simdata.get_SimInfo()
                for info in sim_info:
                    info['condname'] = SimCondList[titlecounter]
                    titlecounter += 1
                SimInfo += sim_info
        else:
            for SimData, condname in zip(SimDataList, SimCondList):
                sim_info = SimData.get_SimInfo()
                for info in sim_info:
                    info['condname'] = condname
                SimInfo += sim_info

    elif dimension == 'time':
        # Check if all datasets have the same sample rate, channel names, and channel positions
        for SimData in SimDataList[1:]:
            assert SimData.get_sample_rate() == sampleRate, "Sample Rates are not the same"
            assert SimData.get_channel_names() == channel_names, "Channel names are not the same"
            assert np.array_equal(SimData.get_channel_positions(), channel_positions), "Channel positions are not the same"
        # Check if all datasets have the same number of trials
        for ind,name in enumerate(dataBucketNames):
            for SimData in SimDataList[1:]:
                assert SimData.get_data(name).shape[0] == SimDataList[0].get_data(name).shape[0], "Number of trials is not the same"
            newdata[ind] = np.concatenate([SimData.get_data(name) for SimData in SimDataList], axis=-1)
        
        # Get all unique keys from the SimInfo objects
        all_keys = set().union(*(SimData.get_SimInfo()[0].keys() for SimData in SimDataList))

        SimInfo = []
        for trial in range(SimDataList[0].get_data(name).shape[0]):
            # Initialize a new dictionary for each trial
            trial_info = {key: [] for key in all_keys}
            for SimData in SimDataList:
                sim_info = SimData.get_SimInfo()[trial]
                # Append the values of the keys in trial_info with the values from the current SimData
                for key in sim_info:
                    trial_info[key].append(sim_info[key])
            # Set 'SwitchTime' to the last time point of the first SimData plus one sample
            trial_info['SwitchTime'] = SimDataList[0].get_time()[-1] + 1/SimDataList[0].get_sample_rate()
            # Set 'condname' to the condition name of the first SimData
            trial_info['condname'] = SimDataList[0].get_SimInfo()[0].get('condname', 'n.a.')
            SimInfo.append(trial_info)
        #update time
        time = np.arange(0+1/sampleRate, (newdata[0].shape[-1]/SimDataList[0].get_sample_rate())+1/sampleRate, 1/sampleRate)
    waveData = wd.WaveData(time=time)
    waveData.set_sample_rate(sampleRate)
    waveData.set_channel_names(channel_names)
    waveData.set_channel_positions(channel_positions)
    waveData.set_simInfo(SimInfo)
    waveData.set_trialInfo([SimInfo["condname"] for SimInfo in waveData.get_SimInfo()])
    for ind,name in enumerate(dataBucketNames):
        dataBucket = wd.DataBucket(newdata[ind], name,dimord, time=time ,chanNames= channel_names)
        waveData.add_data_bucket(dataBucket)
    return waveData
