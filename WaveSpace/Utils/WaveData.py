import pickle
import re

import numpy as np

from . import HelperFuns as hf
from . import ImportHelpers


class DataBucket:
    def __init__(self, data, description, dimord, chanNames, sampleRate=1000, time=None, unit=""):
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
        if time is None:
            time = []
        
        self._data = data
        self._description = description
        self._dimord = dimord
        self._trialInfo = []
        self._chanNames = chanNames
        self._unit = unit
        self._reservedNames = ["time", "chan", "posx", "posy", "trl"]
        if len(time) == 0:
            if "time" not in dimord:
                print(f"Warning: no time dimension in databucket: {self._description} \n timevector will be empty")
                self._time = []
            else:
                totalSamples = self._data.shape[self._dimord.split("_").index("time")]
                totalTimeS = (totalSamples / sampleRate) 
                self._time = np.linspace(0, totalTimeS, num=totalSamples, endpoint=False)
        else:
            self._time = time    

    def get_channel_names(self):
        """Return the channel names stored with this bucket.

        Returns
        -------
        sequence of str
            Channel labels associated with the channel or spatial dimensions.
        """
        return self._chanNames
    
    def get_dimord(self):
        """Return the bucket dimension order.

        Returns
        -------
        str
            Underscore-separated dimension names, such as
            ``"trl_chan_time"``.
        """
        return self._dimord

    def set_dimord(self, dimord):
        """Set the bucket dimension order.

        Parameters
        ----------
        dimord : str
            Underscore-separated dimension names for the stored data.

        Returns
        -------
        None
            Updates the bucket metadata in place.
        """
        self._dimord = dimord

    def get_description(self):
        """Return the bucket name.

        Returns
        -------
        str
            Description used as the bucket key in a WaveData object.
        """
        return self._description

    def set_description(self, description):
        """Set the bucket name.

        Parameters
        ----------
        description : str
            New description used to identify the bucket.

        Returns
        -------
        None
            Updates the bucket metadata in place.
        """
        self._description = description
    
    def get_data(self):
        """Return the data stored in this bucket.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Stored data object without copying it.
        """
        return self._data
    
    def get_time(self):
        """Return the bucket time vector.

        Returns
        -------
        array-like
            Time values associated with the bucket, or an empty list when no
            time dimension exists.
        """
        return self._time
    
    def set_time(self, time):
        """Set the bucket time vector.

        Parameters
        ----------
        time : array-like
            Time values associated with the bucket's time dimension.

        Returns
        -------
        None
            Updates the bucket metadata in place.
        """
        self._time = time
    
    def set_data(self, data, dimord):
        """Replace the stored data and its dimension order.

        Parameters
        ----------
        data : numpy.ndarray or pandas.DataFrame
            Replacement data array.
        dimord : str
            Underscore-separated dimension order for ``data``.

        Returns
        -------
        None
            Replaces the stored data and dimension order in place.

        Raises
        ------
        AssertionError
            If the data rank does not match the number of dimensions in
            ``dimord``.
        """
        assert len(data.shape) == len(dimord.split("_")), "Dimord does not match data dimensions"
        self._dimord = dimord
        self._data = data
        print("Warning: Dangerous move to set data directly buddy, be sure to know what you're doing")
    
    def get_unit(self):
        """Return the physical unit of the bucket data.

        Returns
        -------
        str
            Stored unit label.
        """
        return self._unit
    
    def set_unit(self, unit):
        """Set the physical unit of the bucket data.

        Parameters
        ----------
        unit : str
            Unit label for the stored data.

        Returns
        -------
        None
            Updates the bucket metadata in place.
        """
        self._unit = unit

    def reshape(self, shape, newDimord):
        """Reshape stored data and update its dimension order.

        Parameters
        ----------
        shape : tuple of int
            Target data shape.
        newDimord : str
            Underscore-separated target dimension order.

        Returns
        -------
        None

        Notes
        -----
        Spatial dimensions must be named ``chan`` or begin with ``pos``.
        """
        splitDimord = newDimord.split('_')
        assert len(shape) == len(splitDimord), "Dimensions of new shape do not match dimensions of new dimension order"
        self.set_dimord(newDimord)
        self._data = np.reshape(self._data, shape, order="C")
        chanShape =  tuple([shape[i] for i in  [ind for ind, s in enumerate(splitDimord) if s[0:3]== 'pos']])        
        if len(chanShape) > 0 :
            self._chanNames = np.reshape(self._chanNames, chanShape, order="C")

class WaveData:
    def __init__(self, chanpos=None, coords2D=None, time = None, sampleRate=0.0):
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
        if time is None:
            time = []
        if coords2D is None:
            coords2D = []
        if chanpos is None:
            chanpos = []
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
        """Return a readable summary of stored data buckets.

        Returns
        -------
        str
            Bucket names, dimension orders, data shapes, sampling rate, and
            the active bucket's time range when available.
        """
        out= ""
        for key, dataBucket in self.DataBuckets.items():
            out += f"DataBuckets[\"{key}\"]| {dataBucket.get_dimord()} | {dataBucket.get_data().shape} \n"
        out += "{} | {}(Hz) \n".format("Sampling Rate", self._sampleRate)
        if len(self.get_time()>1):
            out += "{} | {}(S) - {}(S) \n".format("Time", self.get_time()[0], self.get_time()[-1])
        return out  
    
    def append_dataset(self, wavedata, dataBucketName):
        """Append the active data bucket from another WaveData object.

        Parameters
        ----------
        wavedata : WaveData
            Source object.
        dataBucketName : str
            Destination bucket name.

        Returns
        -------
        None
        """
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
        """Return data from the active bucket.

        Returns
        -------
        numpy.ndarray or pandas.DataFrame
            Data stored in the bucket named by ``ActiveDataBucket`` variable.

        Raises
        ------
        KeyError
            If no active bucket is defined.
        """
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
            raise Warning(f"DataBucket {dataBucketName} already exists, overwriting it")
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
        if dataBucketName in self.DataBuckets:
            del self.DataBuckets[dataBucketName]
        else:
            raise NameError("DataBucket does not exist")

    def has_data_bucket(self, bucket_name):
        """Check whether a data bucket exists.

        Parameters
        ----------
        bucket_name : str
            Bucket name.

        Returns
        -------
        bool
            True when the bucket exists.
        """
        return bucket_name in self.DataBuckets

    def crop_data(self, start, stop, dataBucketName=""):
        """Crop a data bucket to a time interval.

        Parameters
        ----------
        start, stop : float
            Requested start and stop times in the units of the bucket time
            vector.
        dataBucketName : str, default=""
            Bucket to crop. An empty string uses the active bucket.

        Returns
        -------
        None
            Replaces the selected bucket's data and time vector with the
            cropped interval and appends a history record.

        Notes
        -----
        The requested bounds are mapped to their nearest time samples.
        """
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
        """Remove selected trials from data buckets and trial metadata.

        Parameters
        ----------
        trials_to_remove : sequence of int
            Trial indices to remove.
        dataBucketName : str or None, default=None
            Name of one bucket to prune. When None, every data bucket with a
            ``trl`` dimension is pruned.

        Returns
        -------
        None
            Updates selected data buckets and removes corresponding entries
            from ``trialInfo``.

        Notes
        -----
        Buckets without a ``trl`` dimension are left unchanged.
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
        """Append an operation record to the processing history.

        Parameters
        ----------
        log : list
            Record containing at least a full method name and shorthand name.

        Returns
        -------
        None
            Appends ``log`` to the internal history list.

        Raises
        ------
        Exception
            If ``log`` contains fewer than two items.
        """
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
        elif isinstance(chanpos, str):
            self._chanpos= ImportHelpers.load_channel_positions(chanpos)
        else:
            raise TypeError("Incorrect format for channel positions. Supply ND-array or filepath")
        
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
        """Set channel names stored on the WaveData object.

        Parameters
        ----------
        ch_names : sequence of str
            Channel labels.

        Returns
        -------
        None
            Updates channel-name metadata in place.
        """
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
        if not (name in self.DataBuckets):
            raise KeyError(f"DataBucket {name} does not exist, can not set as active databucket")
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
        """Set two-dimensional channel coordinates.

        Parameters
        ----------
        coords : numpy.ndarray
            Array with one two-dimensional coordinate per channel.

        Returns
        -------
        None
            Updates two-dimensional coordinate metadata in place.
        """
        self._coords2D = coords

    def set_simInfo(self,simInfo):
        """Set simulation metadata.

        Parameters
        ----------
        simInfo : object
            Metadata describing the simulated data.

        Returns
        -------
        None
            Replaces stored simulation metadata.
        """
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
        with open(filename, 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
        
    def get_SimInfo(self):
        """Return simulation metadata.

        Returns
        -------
        object
            Stored simulation metadata.
        """
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
        """Return the sampling frequency.

        Returns
        -------
        float
            Sampling frequency in Hz.
        """
        return self._sampleRate

    def get_channel_positions(self):
        """Return a copy of channel positions.

        Returns
        -------
        numpy.ndarray
            Copy of the stored channel-position array.
        """
        return np.copy(self._chanpos)

    def get_distMat(self):
        """Return the channel distance matrix.

        Returns
        -------
        numpy.ndarray or list
            Stored pairwise channel-distance matrix.
        """
        return self._distMat
    
    def get_extentGeodesic(self):
        """Return stored geodesic spatial extents.

        Returns
        -------
        tuple of float
            Maximum geodesic distances along the two spatial axes.
        """
        return self._extentGeodesic

    def get_2d_coordinates(self):
        """Return stored two-dimensional channel coordinates.

        Returns
        -------
        numpy.ndarray or list
            Two-dimensional coordinates for each channel.
        """
        return self._coords2D

    def get_log_history(self):
        """Return the processing history.

        Returns
        -------
        list
            Ordered records added through :meth:`log_history`.
        """
        return self._history

    def get_channel_names(self):
        """Return WaveData-level channel names.

        Returns
        -------
        sequence of str
            Stored channel labels.
        """
        return self._channames

    def get_trialInfo(self):
        """Return trial-level labels or metadata.

        Returns
        -------
        sequence
            Stored metadata with one entry per trial when available.
        """
        return self._trialInfo
