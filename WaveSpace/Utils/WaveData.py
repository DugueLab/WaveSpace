from . import ImportHelpers
import numpy as np
from . import HelperFuns as hf
import re
import pickle

class DataBucket:
    def __init__(self, data, description, dimord, chanNames, sampleRate=1000, time=[], unit=""):
        """Store one named data array and its metadata.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            Data stored in the bucket.
        description : str
            Unique bucket name used when adding the bucket to a WaveData
            object.
        dimord : str
            Underscore-separated dimension order, such as ``"trl_chan_time"``
            or ``"trl_posx_posy_time"``.
        chanNames : sequence of str
            Channel names corresponding to the channel or spatial dimensions.
        sampleRate : float, default=1000
            Sampling frequency in Hz used to generate a time vector when
            ``time`` is not supplied.
        time : array-like, default=[]
            Explicit time vector. When omitted and ``dimord`` has a time
            dimension, it is generated from ``sampleRate``.
        unit : str, default=""
            Physical unit of the stored data.
        """
        self._data = data
        self._description = description
        self._dimord = dimord
        self._trialInfo = []
        self._chanNames = chanNames
        self._unit = unit
        self._reservedNames = ["time", "chan", "posx", "posy", "trl"]
        if len(time) == 0:
            if not("time" in dimord):
                print(f"Warning: no time dimension in databucket: {self._description} \n timevector will be empty")
                self._time = []
            else:
                totalSamples = self._data.shape[self._dimord.split("_").index("time")]
                totalTimeS = (totalSamples / sampleRate) 
                self._time = np.linspace(0, totalTimeS, num=totalSamples, endpoint=False)
        else:
            self._time = time    

    def get_channel_names(self):
        return self._chanNames
    
    def get_dimord(self):
        return self._dimord

    def set_dimord(self, dimord):
        self._dimord = dimord

    def get_description(self):
        return self._description

    def set_description(self, description):
        self._description = description
    
    def get_data(self):
        return self._data
    
    def get_time(self):
        return self._time
    
    def set_time(self, time):
        self._time = time
    
    def set_data(self, data, dimord):
        assert len(data.shape) == len(dimord.split("_")), "Dimord does not match data dimensions"
        self._dimord = dimord
        self._data = data
        print("Warning: Dangerous move to set data directly buddy, be sure to know what you're doing")
    
    def get_unit(self):
        return self._unit
    
    def set_unit(self, unit):
        self._unit = unit

    def reshape(self, shape, newDimord):
        """Spatial dimensions must be called chan or pos(char) like posx posy etc.."""
        splitDimord = newDimord.split('_')
        assert len(shape) == len(splitDimord), "Dimensions of new shape do not match dimensions of new dimension order"
        self.set_dimord(newDimord)
        self._data = np.reshape(self._data, shape, order="C")
        chanShape =  tuple([shape[i] for i in  [ind for ind, s in enumerate(splitDimord) if s[0:3]== 'pos']])        
        if len(chanShape) > 0 :
            self._chanNames = np.reshape(self._chanNames, chanShape, order="C")

    def assure_consistency():
        return None

class WaveData():
    def __init__(self, chanpos=[], coords2D=[], time = [], sampleRate=0.0):
        """Create an empty container for WaveSpace data buckets and metadata.

        Parameters
        ----------
        chanpos : array-like, default=[]
            Initial channel positions, conventionally an array with one row per
            channel and three spatial coordinates.
        coords2D : array-like, default=[]
            Reserved argument for initial two-dimensional coordinates.
        time : array-like, default=[]
            Reserved argument for initial time values. Time is normally stored
            on individual DataBucket objects.
        sampleRate : float, default=0.0
            Sampling frequency in Hz for the data collection.

        Notes
        -----
        Add data through :meth:`add_data_bucket`. The latest added bucket
        becomes the active data bucket.
        """
        self.DataBuckets= {}
        self.ActiveDataBucket = ""
        self.HasRegularLayout = False

        self._sampleRate = sampleRate
        self._chanpos = chanpos
        self._simInfo = []
        self._trialInfo = []
        self._history = []
        self._distMat = []
        self._coords2D = []
        self._channames =[]
        

    def __repr__(self):
        out= ""
        for key, dataBucket in self.DataBuckets.items():
            out += "DataBuckets[\"%s\"]| %s | %s \n" % (key , dataBucket.get_dimord(),dataBucket.get_data().shape )
        out += "%s | %s(Hz) \n" % ("Sampling Rate", self._sampleRate)
        if len(self.get_time()>1):
            out += "%s | %s(S) - %s(S) \n" % ("Time", self.get_time()[0], self.get_time()[-1])
        return out  
    
    def append_dataset(self, wavedata, dataBucketName):
        """Appends active bucket of the supplied wavedata-object to databucket with dataBucketName in the current wavedata"""
        data = wavedata.DataBuckets[wavedata.ActiveDataBucket].get_data()
        self.DataBuckets[dataBucketName]._data = np.concatenate([self.DataBuckets[dataBucketName]._data, data], axis=0)
        self._simInfo += wavedata.get_SimInfo()

    def get_data(self, name):
        """Return data from a named bucket.

        Parameters
        ----------
        name : str
            Name of the data bucket.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Data stored in the requested bucket.
        """
        return self.DataBuckets[name].get_data()

    def get_active_data(self):
        return self.DataBuckets[self.ActiveDataBucket].get_data()

    def add_data_bucket(self, dataBucketName):
        """Add a DataBucket and make it the active data bucket.

        Parameters
        ----------
        dataBucketName : DataBucket
            Bucket to store, using its description as the dictionary key.

        Returns
        -------
        None
        """
        if (self.has_data_bucket(dataBucketName)):
            Warning(f"DataBucket {dataBucketName} already exists, overwriting it")
        name = dataBucketName.get_description()
        self.ActiveDataBucket = name
        self.DataBuckets[name] = (dataBucketName)

    def delete_data_bucket(self, dataBucketName):
        """Remove a named data bucket.

        Parameters
        ----------
        dataBucketName : str
            Name of the bucket to remove.

        Returns
        -------
        None

        Raises
        ------
        NameError
            If no bucket has the requested name.
        """
        if dataBucketName in self.DataBuckets.keys():
            del self.DataBuckets[dataBucketName]
        else:
            raise NameError("DataBucket does not exist")

    def has_data_bucket(self, bucket_name):
        '''Check if a data bucket with a bucket_name exists in the WaveData object.
            bucket_name : str
            The name of the data bucket to check
            Returns 
            exists : bool
            True if the data bucket exists, False otherwise
        '''
        return bucket_name in self.DataBuckets

    def crop_data(self, start, stop, dataBucketName=""):
        if  dataBucketName == "":
            dataBucketName =  self.ActiveDataBucket
        else:
            self.set_active_dataBucket(dataBucketName)

        t0,_=hf.find_nearest(self.get_time(dataBucketName), start)#Index of start time of interest
        t1,_=hf.find_nearest(self.get_time(dataBucketName), stop)#Index of end time of interest 
        self.set_time(self.get_time()[t0:t1])
        dimensions = self.DataBuckets[self.ActiveDataBucket]._dimord.split("_")
        timedim = [ind for ind, item in enumerate(dimensions) if re.search("time", item)]
        self.DataBuckets[self.ActiveDataBucket]._data = self.DataBuckets[self.ActiveDataBucket]._data.take(indices=range(t0,t1), axis = timedim[0])
        self.log_history(["Crop", "Start",t0, "Stop", t1])  
    
    def prune_trials(self, trials_to_remove, dataBucketName=None):
        """Prune trials from the data and trialInfo list. 
        If dataBucketName is None, prunung is done from on all data buckets!!!
        Args:
            trials_to_remove (list): A list of trial indices to remove.
            dataBucketName (str, optional): Name of the data bucket to prune. If None, prune all.
        """
        buckets = [dataBucketName] if dataBucketName else list(self.DataBuckets.keys())
        for bucket in buckets:
            dimensions = self.DataBuckets[bucket]._dimord.split("_")
            trialdim = [ind for ind, item in enumerate(dimensions) if re.search("trl", item)]
            if trialdim:
                self.DataBuckets[bucket]._data = np.delete(self.DataBuckets[bucket]._data, trials_to_remove, axis=trialdim[0])
                print(f'Pruned {len(trials_to_remove)} trials from dataBucket {bucket}')
                print(f'New data shape: {self.DataBuckets[bucket]._data.shape}')
        self._trialInfo = [trial for i, trial in enumerate(self._trialInfo) if i not in trials_to_remove]

    def log_history(self, log):
        if (not(len(log) >= 2)):
            raise Exception("Input to log requires a list with at least two string items (Full name & shorthand of method)")
        else:
            self._history.append(log)       
    
    def set_channel_positions(self, chanpos):
        """Set three-dimensional channel positions.

        Parameters
        ----------
        chanpos : numpy.ndarray or str
            Channel-position array or path to a serialized channel-position
            file.

        Returns
        -------
        None
        """
        if (type(chanpos) == np.ndarray):
            self._chanpos = chanpos
        elif type(chanpos) == str:
            self._chanpos= ImportHelpers.load_channel_positions(chanpos)
        else:
            raise Exception("Incorrect format for channel positions. Supply ND-array or filepath")
        
    def set_time(self, time, dataBucketName = ""):
        """Set the time vector for a data bucket.

        Parameters
        ----------
        time : array-like
            Time values to assign.
        dataBucketName : str, default=""
            Bucket to update. By default, the active data bucket is used.

        Returns
        -------
        None
        """
        if dataBucketName == "":
            dataBucketName = self.ActiveDataBucket
        self.DataBuckets[dataBucketName].set_time(time)

    def set_channel_names(self, ch_names):
        self._channames = ch_names

    def set_active_dataBucket(self, name):
        """Select the active data bucket.

        Parameters
        ----------
        name : str
            Name of an existing data bucket.

        Returns
        -------
        None

        Raises
        ------
        Exception
            If no bucket has the requested name.
        """
        if not (name in self.DataBuckets.keys()):
            raise Exception(f"DataBucket {name} does not exist, can not set as active databucket")
        self.ActiveDataBucket = name

    def set_sample_rate(self, sampleRate):
        """Set the sampling frequency.

        Parameters
        ----------
        sampleRate : float
            Sampling frequency in Hz.

        Returns
        -------
        None
        """
        self._sampleRate = sampleRate

    def set_distMat(self, distMat):
        """Set the channel distance matrix.

        Parameters
        ----------
        distMat : numpy.ndarray
            Square matrix of pairwise channel distances.

        Returns
        -------
        None
        """
        self._distMat = distMat

    def set_2D_coordinates(self, coords):
        self._coords2D = coords

    def set_simInfo(self,simInfo):
        self._simInfo = simInfo

    def set_trialInfo(self, trialInfo):
        """Set trial-level labels or metadata.

        Parameters
        ----------
        trialInfo : sequence
            One label or metadata item per trial.

        Returns
        -------
        None
        """
        self._trialInfo = trialInfo

    def save_to_file(self, filename=""):
        """Serialize this WaveData object to a pickle file.

        Parameters
        ----------
        filename : str or path-like, default=""
            Output path. By default, a name is generated from the history log.

        Returns
        -------
        None
        """
        if filename=="":
            filename = "WaveData_" + '_'.join([element[1] for element in self._history])
        
        f = open(filename, 'wb')
        pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        
    def get_SimInfo(self):
        return self._simInfo

    def get_time(self, dataBucketName=""):
        """Return a data bucket's time vector.

        Parameters
        ----------
        dataBucketName : str, default=""
            Name of the bucket. By default, the active data bucket is used.

        Returns
        -------
        array-like
            Time values stored in the selected bucket.
        """
        if dataBucketName == "":
            return self.DataBuckets[self.ActiveDataBucket].get_time()
        return self.DataBuckets[dataBucketName].get_time()

    def get_sample_rate(self):
        """Return the sampling frequency in Hz."""
        return self._sampleRate

    def get_channel_positions(self):
        """Return a copy of the channel-position array."""
        return np.copy(self._chanpos)

    def get_distMat(self):
        """Return the channel distance matrix."""
        return self._distMat
    
    def get_extentGeodesic(self):
        return self._extentGeodesic

    def get_2d_coordinates(self):
        """Return the two-dimensional channel coordinates."""
        return self._coords2D

    def get_log_history(self):
        return self._history

    def get_channel_names(self):
        return self._channames

    def get_trialInfo(self):
        """Return the trial-level labels or metadata."""
        return self._trialInfo
