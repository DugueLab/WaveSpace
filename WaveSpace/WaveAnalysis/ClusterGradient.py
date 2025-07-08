from joblib import Parallel, delayed
import WaveSpace.Utils.HelperFuns as hf
import WaveSpace.Utils.WaveData as wd
import numpy as np


def perform_cluster_gradient(waveData, dataBucket="FrequencyCluster"):
    waveData.set_active_data_bucket(dataBucket)
    hf.assure_consistency(waveData)
    ClusterContacts = waveData.get_data(dataBucket)
    sampleRate = waveData.get_sample_rate()
    nTrials, nChannels, nTime = ClusterContacts.shape()

    complexData_FreqCluster = waveData.get_data(dataBucket)
    chanpos2D = chanpos[:,:2]
    # For real data some sort of projection has to happen there e.g. 
    # project the first 2 principle components: 
    # pca = decomposition.PCA(n_components=2)
    # chanpos2D=pca.fit_transform(chanpos)
    for trl in range(nTrials):
        for cluster in range(complexData_FreqCluster[trl].shape[0]):
            ClusterPhase = np.angle(complexData_FreqCluster[trl][cluster])
            clusterchans = ClusterContacts[trl][cluster]
            angle_SF_corr_offset = getClusterGradient(ClusterPhase, chanpos2D[clusterchans,:]) 
    angle_bucket = wd.DataBucket(angle_SF_corr_offset, "Angle_sf", "trl_chan_freqcluster_time")
    waveData.add_data_bucket(angle_bucket)

def getClusterGradient(ClusterPhase, chanpos2D):
 #complexData contains data per trial and frequency cluster (i.e., each [trl][Freqcluster] has dimord [channel,time])
 #chanpos is the 2D or 3D positions of all channels within the current cluster 
    #get the maximum distance between neighboring contacts:
    distances = []
    for ii in range(len(chanpos2D)):
        #all distances between closest neighbors (within this cluster):
        distances.append(np.min(np.linalg.norm(np.concatenate((chanpos2D[:ii,:],chanpos2D[ii+1:,:]))-chanpos2D[ii,:], axis=1)))
    spatial_resolution = np.max(distances) #max spacing between neighboring contacts. [KP]What about irregular sampling here?

    thetas = np.radians(np.arange(0, 356, 5))
    rs = np.radians(np.arange(0, 180/spatial_resolution, .3)) #take the spatial nyquist into account 
    theta_rs = np.stack([(x, y) for x in thetas for y in rs])
    params = np.stack([theta_rs[:, 1] * np.cos(theta_rs[:, 0]), theta_rs[:, 1] * np.sin(theta_rs[:, 0])], -1)

    #fit 2D circular linear model of phase across all positions at each timepoint: 
    data_as_list = zip(np.expand_dims(ClusterPhase.T,1),np.array([chanpos2D]), [theta_rs], [params])
    res_as_list = Parallel(n_jobs=40, verbose=0)(delayed(circ_lin_regress)(x[0], x[1], x[2], x[3]) for x in data_as_list)
    local_angle = res_as_list[0][0]
    local_sf    = res_as_list[0][1]
    local_corr    = res_as_list[0][2]
    local_offset   = res_as_list[0][3]
    return local_angle, local_sf, local_corr, local_offset

def circ_lin_regress(phases, coords, theta_r, params):
    import pycircstat
    """
    Performs 2D circular linear regression.
    This is from https://github.com/john-myers-github/INSULA_RS, and was originally ported from Honghui's matlab code.
    """

    n = phases.shape[1]
    pos_x = np.expand_dims(coords[:, 0], 1)
    pos_y = np.expand_dims(coords[:, 1], 1)

    # compute predicted phases for angle and phase offset
    x = np.expand_dims(phases, 2) - params[:, 0] * pos_x - params[:, 1] * pos_y
   
    # Compute resultant vector length. 
    x1 = np.sum(np.cos(x) / n, axis=1)
    x1 = x1 ** 2
    x2 = np.sum(np.sin(x) / n, axis=1)
    x2 = x2 ** 2
    Rs = -np.sqrt(x1 + x2)


    # for each time and event, find the parameters with the smallest -R
    min_vals = theta_r[np.argmin(Rs, axis=1)]

    sl = min_vals[:, 1] * np.array([np.cos(min_vals[:, 0]), np.sin((min_vals[:, 0]))])
    offs = np.arctan2(np.sum(np.sin(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0),
                      np.sum(np.cos(phases.T - sl[0, :] * pos_x - sl[1, :] * pos_y), axis=0))
    pos_circ = np.mod(sl[0, :] * pos_x + sl[1, :] * pos_y + offs, 2 * np.pi)

    # compute circular correlation coefficient between actual phases and predicited phases
    circ_corr_coef = pycircstat.corrcc(phases.T, pos_circ, axis=0)

    # compute adjusted r square
    # pdb.set_trace()
    r2_adj = circ_corr_coef ** 2
    #r2_adj = 1 - ((1 - circ_corr_coef ** 2) * (n - 1)) / (n - 4)

    wave_ang = min_vals[:, 0]
    wave_freq = min_vals[:, 1]
    return wave_ang, wave_freq, r2_adj, offs