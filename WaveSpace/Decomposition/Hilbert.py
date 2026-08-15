import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd
from scipy.signal import hilbert
import numpy as np

def apply_hilbert(waveData, dataBucketName =None ):
    """Compute the analytic signal with a Hilbert transform.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the data to transform.
    dataBucketName : str, default=None
        Name of the input data bucket. Defaults to the active data bucket.

    Returns
    -------
    None
        Adds the complex analytic signal to ``waveData`` as the ``complexData``
        databucket.

    Notes
    -----
    The transform is computed along the time dimension. For interpretable
    instantaneous phase, use a sufficiently narrowband input such as the
    output of :func:`WaveSpace.Preprocessing.Filter.filter_narrowband`.
    """
    # ensure proper bookkeeping of data dimensions
    if dataBucketName == None:
        dataBucketName = waveData.ActiveDataBucket
    else:
        waveData.set_active_dataBucket(dataBucketName)
    #check if any filtereing has been done before, if so, warn that it will be overwritten
    if not (dataBucketName == "NBFiltered"):
        print("Hilbert Transform to get analytic signal: Make sure data has been filtered to a sufficiently narrow frequency bandwith before applying hilbert")

    hf.assure_consistency(waveData)
    currentData = waveData.DataBuckets[dataBucketName].get_data()
    origDimord = waveData.DataBuckets[dataBucketName].get_dimord()
    origShape = currentData.shape
    hasBeenReshaped, currentData =  hf.force_dimord(currentData, origDimord , "trl_chan_time")
    # Compute Hilbert transform
    analytic_signal = hilbert(currentData)
    inst_amplitude = np.abs(analytic_signal)
    inst_phase  = np.unwrap(np.angle(analytic_signal))
    complexData = inst_amplitude * np.exp(1j * inst_phase)
    if hasBeenReshaped:
        complexData = np.reshape(complexData, origShape)

    complexDataBucket = wd.DataBucket(complexData, "complexData", origDimord,time=waveData.DataBuckets[waveData.ActiveDataBucket].get_time(),chanNames= waveData.get_channel_names())
    waveData.add_data_bucket(complexDataBucket)
    waveData.log_history(["Analytic Signal", "Hilbert"])




