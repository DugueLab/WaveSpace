import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import pyvista as pv
from matplotlib import animation, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from plotly.subplots import make_subplots
from scipy.interpolate import griddata

import WaveSpace.Utils.HelperFuns as hf


def init():
    """
    Returns
    -------
    None
    """
    plt.style.use("settings.mplstyle")

def getProbeColor(index, totalProbes, cmap = plt.cm.ocean):
    """Return a colormap color for a selected probe.

    Parameters
    ----------
    index : int
        Zero-based index of the probe.
    totalProbes : int
        Total number of probes used to distribute colors across the colormap.
    cmap : matplotlib.colors.Colormap, default=matplotlib.pyplot.cm.ocean
        Colormap from which to select the color.

    Returns
    -------
    tuple of float
        RGBA color for the requested probe.
    """
    return cmap(index/totalProbes)

def get_color_grid_from_probes(gridsize, probes):
    """Create an RGBA grid that marks selected spatial probe positions.

    Parameters
    ----------
    gridsize : int or tuple of int
        Square grid size or ``(rows, columns)`` grid shape.
    probes : sequence of tuple of int
        ``(row, column)`` grid positions to color. Positions not in this list
        are grey.

    Returns
    -------
    numpy.ndarray
        RGBA color array with shape ``(rows, columns, 4)``.
    """
    if isinstance(gridsize, int):
        rows, cols = gridsize, gridsize
    else:
        rows, cols = gridsize
    rows * cols
    total_probes = len(probes)

    # Fill the grid row-wise with probe colors, repeating or truncating as needed
    color_grid = np.zeros((rows, cols, 4))  # RGBA shape
    for i in range(rows):
        for j in range(cols):
            if (i, j) in probes:                
                color = getProbeColor(probes.index((i,j)), total_probes)
            else:
                color = (0.5, 0.5, 0.5, 1.0)  # Transparent or black if no more probes
            color_grid[i, j] = color
    return color_grid

def add_color_grid_legend(ax, color_grid, position=None, border=True):
    """Add a color-grid legend as an inset in a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes that receive the inset legend.
    color_grid : numpy.ndarray
        RGB or RGBA color array with grid dimensions first.
    position : sequence of float, default=[0.8, 0.8, 2.0, 2.0]
        Inset left and bottom positions in axes coordinates, followed by its
        width and height.
    border : bool, default=True
        Show the inset axes border.

    Returns
    -------
    matplotlib.axes.Axes
        The inset axes containing the color grid.
    """
    # Ensure the grid is in shape (rows, cols, 4)
    if position is None:
         position = [0.8, 0.8, 2.0, 2.0]
    _rows, _cols = color_grid.shape[:2]

    # Create an inset axis
    inset_ax = inset_axes(ax, width=position[2], height=position[3],
                          bbox_to_anchor=(position[0], position[1], 1, 1),
                          bbox_transform=ax.transAxes, borderpad=0)

    # Use imshow to draw the grid
    inset_ax.imshow(color_grid, aspect='equal', interpolation='none', origin='lower' )

    # Hide ticks and spines
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_visible(border)

    return inset_ax
 
def plotfft_zoomed(fft_abs, sfreq, minFreq, maxFreq, title, scale='linear'):    
    """Plot selected temporal frequencies from a spatio-temporal FFT.

    Parameters
    ----------
    fft_abs : numpy.ndarray
        Two-dimensional FFT power array ordered as spatial frequency by
        temporal frequency.
    sfreq : float
        Sampling frequency in Hz.
    minFreq : float
        Lower temporal-frequency display bound in Hz.
    maxFreq : float
        Upper temporal-frequency display bound in Hz.
    title : str
        Title prefix for the plot.
    scale : {"linear", "log"}, default="linear"
        Power display scale. ``"log"`` applies log10 scaling and normalizes
        the plotted values.

    Returns
    -------
    module
        The ``matplotlib.pyplot`` module containing the created plot.
    """
    nChan, nTimepoints = fft_abs.shape
    spatialFreqAxis = nChan/2 * np.linspace(-1, 1, nChan)
    tempFreqAxis = np.arange(-sfreq/2, sfreq/2, 1/(nTimepoints/sfreq))
    plotrange = np.where((tempFreqAxis > minFreq) & (tempFreqAxis < maxFreq))
    if scale == 'log':
        fft_abs = np.log10(fft_abs + 1e-12)
        fft_abs = (fft_abs - np.min(fft_abs)) / (np.max(fft_abs) - np.min(fft_abs))

    plt.imshow(fft_abs[:, plotrange[0]], aspect="auto", extent=[tempFreqAxis[plotrange[0]][0], tempFreqAxis[plotrange[0]][-1], spatialFreqAxis[0], spatialFreqAxis[-1]])
    plt.colorbar(label="Power (dB)" if scale == "log" else "Power")
    plt.title(f"{title} Spatial Freq over Temporal Freq")
    plt.xlabel("Temporal Frequency (Hz)")
    plt.ylabel("Spatial Frequency (channels/Hz)")
    #plt.show()
    return plt  

def plot_imfs(waveData, dataInds = (0), IMFofInterest = 1):
    """Plot intrinsic mode functions and the phase of one selected IMF.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing EMD output in its ``complexData`` bucket.
    dataInds : tuple, default=(0,)
        Indices selecting one IMF-by-time series from ``complexData``.
    IMFofInterest : int, default=1
        Zero-based IMF index whose phase is shown in the second figure.

    Returns
    -------
    imf_figure : matplotlib.figure.Figure
        Figure produced by ``emd.plotting.plot_imfs``.
    phase_figure : matplotlib.figure.Figure
        Figure showing the selected IMF phase over time.
    """
    import emd
    time = waveData.get_time()
    imfs = waveData.get_data("complexData")[dataInds]    
    imfs = imfs.T
    IP = np.angle(imfs[:,IMFofInterest])  
    # remove any imfs that are NaN
    imfs = imfs[:,~np.isnan(imfs[0,:])]    
    emd.plotting.plot_imfs(imfs=imfs, time_vect=time, cmap=True, xlabel = 'Time (seconds)')
    f1 = plt.gcf()
    f2 = plt.figure(figsize= [16, 3])
    # Plot Phase
    plt.plot(time,IP)
    plt.title('Phase of IMF of Interest (IMF '+ str(IMFofInterest+1))
    plt.xlabel('Time (seconds)')
    plt.xlim(time[0], time[-1])
    plt.yticks(np.arange(-np.pi, np.pi, step=np.pi/2), [r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in np.arange(-np.pi, np.pi, step=np.pi/2)])
    plt.ylim(-np.pi, np.pi)
    plt.ylabel('Phase')
    return f1 , f2
    #plt.subplots_adjust(left=0.4, right=0.99)

def plot_interpolated_data(waveData, original_data_bucket, interpolated_data_bucket, grid_x, grid_y, OrigInd, InterpInd, type = ""):
    """Compare original sensor data with an interpolated grid.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing original and interpolated data buckets.
    original_data_bucket : str
        Name of the original sensor-data bucket.
    interpolated_data_bucket : str
        Name of the interpolated grid-data bucket.
    grid_x, grid_y : numpy.ndarray
        Two-dimensional coordinate arrays e.g. returned by
        :func:`interpolate_pos_to_grid`.
    OrigInd : tuple
        Indices selecting a channel-value vector from the original bucket.
    InterpInd : tuple
        Indices selecting a spatial grid from the interpolated bucket.
    type : {"phase", "angle", "power", "abs"}, default=""
        Display transformation. The default displays raw values; phase and
        power modes display complex angle and magnitude, respectively.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing original three-dimensional positions, projected
        two-dimensional positions, and interpolated-grid values.
    """
    original_data = waveData.get_data(original_data_bucket)[OrigInd]
    interpolated_data = waveData.get_data(interpolated_data_bucket)[InterpInd].ravel()
    norm = None #default colormapping

    if type == "phase" or type == "angle":
        original_data = np.angle(original_data)
        interpolated_data = np.angle(interpolated_data)
        norm = colors.Normalize(vmin=-np.pi, vmax=np.pi) #fix range from -pi to pi
    elif type == "power" or type == "abs":
        original_data = np.abs(original_data)
        interpolated_data = np.abs(interpolated_data)
    # if none of the above, just plot the data. If complex this defaults to the real part anyways
    #get 3d positions
    pos_3d = waveData.get_channel_positions()
    # Scale the pos_2d coordinates if needed
    pos_2d = waveData.get_2d_coordinates() 

    #create scatter plot of original 3d channel positions

    fig = plt.figure(figsize=(15, 5))

    # Create scatter plot of original 3D channel positions
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    scatter = ax.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], c=original_data, norm=norm)
    plt.colorbar(scatter, label=type)
    plt.title('Original Data')
    ax.set_xlabel('X coordinate (cm)')
    ax.set_ylabel('Y coordinate (cm)')
    ax.set_zlabel('Z coordinate (cm)')
    ax.view_init(elev=90, azim=-90)  # View the plot from the top
    plt.axis('auto')

    # Create scatter plot of 2D projected positions
    ax2 = plt.subplot(1, 3, 2)
    plt.scatter(pos_2d[:, 0], pos_2d[:, 1], c=original_data, norm=norm)
    plt.colorbar(label=type)
    plt.title('2D Projected Data')
    plt.xlabel('X coordinate (cm)')
    plt.ylabel('Y coordinate (cm)')
    ax2.set_aspect('auto')  # Set the aspect ratio to be equal

    # Create scatter plot of interpolated data
    ax3 = plt.subplot(1, 3, 3)
    plt.scatter(grid_x.ravel(), grid_y.ravel(), c=interpolated_data, norm=norm)
    plt.colorbar(label=type)
    plt.title('Interpolated Data')
    plt.xlabel('X coordinate (cm)')
    plt.ylabel('Y coordinate (cm)')
    ax3.set_aspect('auto')  # Set the aspect ratio to be equal

    plt.tight_layout()
    plt.show()
    return fig

def plot_timeseries_on_surface(Surface, waveData, dataBucketName = " ", indices = (0, 0, None, slice(None), slice(None)), chan_to_highlight = 0 , timepoint =0, plottype = "power"):
    """Plot topo time series on a surface
    + actual timeseries of a selected channel.

    Parameters
    ----------
    Surface : list of numpy.ndarray
        Surface vertices and triangular faces.
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing sensor data and channel positions.
    dataBucketName : str, default=" "
        Name of the data bucket to plot. A single space selects the active
        bucket.
    indices : tuple
        Explicit integer indices select dimensions, ``None`` averages a
        dimension, and ``slice(None)`` retains it.
    chan_to_highlight : int, default=0
        Channel whose time series is plotted and highlighted on the surface.
    timepoint : int, default=0
        Retained for compatibility; slider steps determine the displayed time.
    plottype : {"power", "real", "phase"}, default="power"
        Transformation applied before plotting.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure with a time slider and selected-channel trace.

    Notes
    -----
    After indexing and averaging, data must reduce to channel by time.
    """

    if dataBucketName == " ":
        dataBucketName = waveData.ActiveDataBucket
    waveData.set_active_dataBucket(dataBucketName)
    hf.assure_consistency(waveData)
    dimord= waveData.DataBuckets[waveData.ActiveDataBucket].get_dimord()
    dimord.split("_")
    data = waveData.DataBuckets[waveData.ActiveDataBucket].get_data()

    faces = Surface[1]
    faces = faces.reshape(-1, 3)
    channel_positions = waveData.get_channel_positions()
    time = waveData.get_time()

    # Create base figure
    fig = make_subplots(rows=2, cols=1, specs=[[{'type': 'scene'}], [{'type': 'xy'}]], 
                        subplot_titles=('3D Surface', 'Time Series'), vertical_spacing=0.3)
    
    # Get the channel data. All exlicit indeces are used as such, None is averaged over and slice(None) stays as is
    average_axes = tuple(i for i, index in enumerate(indices) if index is None)
    data = np.mean(data, axis=average_axes, keepdims=True)
    # Create a new set of indices that only includes the dimensions that weren't averaged
    new_indices = tuple(index if isinstance(index, int) else slice(None) for index in indices)
    channel_data = data[new_indices]
    channel_data = channel_data.squeeze()

    # Add traces, one for each slider step
    if plottype == "power":
        channel_data = np.abs(channel_data)
    elif plottype == "real":
        channel_data = np.real(channel_data)
    elif plottype == "phase":
        channel_data = np.angle(channel_data)
    clim = [np.min(channel_data), np.max(channel_data)]
    for timepoint_ in range(len(time)):
        channel_data_snapshot = channel_data[:, timepoint_]
        # Add a trace for the surface
        fig.add_trace(
            go.Mesh3d(
                x=Surface[0][:, 0],
                y=Surface[0][:, 1],
                z=Surface[0][:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color='lightgrey',
                opacity=.8,
                visible=False
            ),
            row=1, col=1
        )
        # Add a trace for the channel positions
        fig.add_trace(
            go.Scatter3d(
                x=channel_positions[:, 0],
                y=channel_positions[:, 1],
                z=channel_positions[:, 2],
                mode='markers',
                marker={"size": 10, "color": channel_data_snapshot, 
                            "cmin": clim[0], "cmax": clim[1],
                            "colorscale": 'RdBu_r', 
                            "colorbar": {"title": plottype, "x": -0.07, "len": 0.7}},
                visible=False
            ),
            row=1, col=1
        )

        # Add a trace to highlight the selected channel
        fig.add_trace(
            go.Scatter3d(
                x=[channel_positions[chan_to_highlight, 0]],
                y=[channel_positions[chan_to_highlight, 1]],
                z=[channel_positions[chan_to_highlight, 2]],
                mode='markers',
                marker={
                    "size": 12, 
                    "color": 'rgba(0,0,0,0)',  # transparent fill
                    "line": {"color": 'red', "width": 5}  # black border
                },
                visible=False
            ),
            row=1, col=1
        )

    
    # Add the time series trace to the current step
    fig.add_trace(
        go.Scatter(
            x=time,
            y=channel_data[chan_to_highlight, :],
            mode='lines',
            line={"color": 'black', "width": 2},
            visible=True
        ),
        row=2, col=1
    )

    # Create and add slider
    steps = []
    for i in range(0, len(fig.data)-1, 3):  
        step = {
            "method": "update",
            "args": [{"visible": [False] * len(fig.data)},  # Start by making all traces invisible
                {"title": "Time: " + str(time[i//3])}],  # layout attribute
        }
        # Make the current 3D traces visible
        if i < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i] = True
        if i+1 < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i+1] = True
        if i+2 < len(step["args"][0]["visible"]):
            step["args"][0]["visible"][i+2] = True

        # Make the time series trace visible
        step["args"][0]["visible"][-1] = True

        # Update the position of the vertical line with the slider
        step["args"].append({"shapes": [
            {
                "type": "line",
                "xref": "x",
                "yref": "paper",
                "x0": time[i//3],
                "y0": 0,
                "x1": time[i//3],
                "y1": 1,
                "line": {
                    "color": "Red",
                    "width": 3
                }
            }
        ]})

        steps.append(step)

    sliders = [{
        "active": 0,
        "currentvalue": {"prefix": "Time: "},
        "pad": {"t": 50},
        "steps": steps
    }]

    fig.update_layout(
        sliders=sliders,
        scene={
            "xaxis": {"nticks": 4, "range": [np.min(Surface[0]),np.max(Surface[0])],},
            "yaxis": {"nticks": 4, "range": [np.min(Surface[0]),np.max(Surface[0])],},
            "zaxis": {"nticks": 4, "range": [np.min(Surface[0]),np.max(Surface[0])],},
            "aspectmode": 'cube',
            "domain": {"y": [0.3, 1]}  # Adjust the size of the 3D subplot
        },
        xaxis={"domain": [0, 1], "anchor": 'y2'},  # Adjust the size of the 2D subplot
        yaxis={"domain": [0, 0.25], "anchor": 'x2'},  # Adjust the size of the 2D subplot
        width=700,
        margin={"r": 20, "l": 10, "b": 10, "t": 10}
    )
    fig.show()
    return fig

def animate_grid_data(gridData,DataBucketName = "", dataInd = None, probepositions=None, plottype = "real"):
    """Animate gridData over time and show time series at selected probe positions.

    Parameters
    ----------
    gridData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing a spatial grid over time.
    DataBucketName : str, default=""
        Name of the input data bucket. Defaults to the active bucket.
    dataInd : int or tuple or None, default=None
        Indices selecting one three-dimensional ``posx_posy_time`` array.
    probepositions : sequence of tuple of int, default=[(0, 0)]
        Spatial probe positions shown on the grid and in the time-series plot.
    plottype : {"real", "power", "angle", "isPhase"}, default="real"
        Transformation applied to the input data before animation.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation of grid values and probe time series.
    """
    if probepositions is None:
         probepositions = [(0, 0)]
    if DataBucketName == "":
        DataBucketName = gridData.ActiveDataBucket
    timevec = gridData.get_time()    

    plotGridSize = (1,2)
    plt.rcParams["figure.autolayout"] = True
    plt.figure(figsize=(plotGridSize[1]*8, plotGridSize[0]*8))

    #IMSHOW grid
    ax1 = plt.subplot2grid(plotGridSize, (0, 0), colspan=1, rowspan=1)
    ax1.grid(None)
    plt.set_cmap('copper')  
    #plt.tight_layout()
    ax1.axis('off')
    
    if dataInd is not None:
        if isinstance(dataInd, int):
            dataToPlot = gridData.get_data(DataBucketName)[dataInd, :, :, :]  #just pick trl
        elif isinstance(dataInd, tuple):
            dataToPlot = gridData.get_data(DataBucketName)[dataInd]  
        else:
            raise ValueError("dataInd must be an integer or a tuple")
    else:
        dataToPlot = gridData.get_data(DataBucketName)
    
    # somehow python sometimes shifts dims around. Check and fix
    if dataToPlot.ndim == 3:
        posx, posy, time = dataToPlot.shape if dataToPlot.shape[1] != dataToPlot.shape[2] else dataToPlot.shape[::-1]#this only works if posx==posy. Fix later
        if dataToPlot.shape == (time, posx, posy):  
            dataToPlot = np.transpose(dataToPlot, (1, 2, 0))
            #if that happens, most likely the timevec has changed
            timevec = timevec[dataInd[-1]]
        elif dataToPlot.shape == (posx, posy, time):  
            pass
        else:
            raise ValueError("dataToPlot does not have the right shape.")
    else:
        raise ValueError(f"dataToPlot should have 3 dimensions after indexing, but got {dataToPlot.ndim}")


    if plottype== 'real':
        dataToPlot = np.real(dataToPlot)
    elif plottype == 'power':
        dataToPlot = np.abs(dataToPlot)
    elif plottype == 'angle': 
        dataToPlot = np.angle(dataToPlot)
    elif plottype == 'isPhase':
        print('data is assumed to already be phase data')
        
    vmin, vmax = np.percentile(dataToPlot, [1, 99])
    img = ax1.imshow(dataToPlot[ :, :, 0],
                    origin='lower', vmin=vmin, vmax=vmax, cmap="copper")

    cbar = plt.colorbar(img)
    cbar.set_label(r'$\mu$V')
    dataToPlot.shape[-1]  
    lengthOfMatrix =  dataToPlot.shape[0] * dataToPlot.shape[1]
    # make all black
    probecolors = []
    [(0.0, 0.0, 0.0)for i in range(lengthOfMatrix)]
    for ind, probe in enumerate(probepositions):
        currentColor = getProbeColor(ind, len(probepositions))
        currentRect = plt.Rectangle((probe[1]-0.5, probe[0]-0.5), 1, 1, facecolor='none',edgecolor=currentColor,lw=2)
        probecolors.append(currentRect.get_edgecolor())
        ax1.add_patch(currentRect)

    currentShape = dataToPlot.shape

    nframes = currentShape[2]
    lineseriesdata = np.zeros((len(probepositions), nframes), dtype='float64')
    currentPlot = plt.subplot2grid(
        plotGridSize, (0,1), colspan=1, rowspan=1)
    currentPlot.plot(timevec, lineseriesdata.T,linewidth=3)
    currentPlot.grid(visible = False)
    currentPlot.set_ylabel([])
    currentPlot.set_facecolor("white")

    ylim = np.array([np.min(dataToPlot), np.max(dataToPlot)])

    linedistance = 2
    if plottype == 'angle' or plottype == 'isPhase':
        img = ax1.imshow(dataToPlot[:, :, 0], origin='lower',vmin=-np.pi, vmax=np.pi)
    else:
        img =  ax1.imshow(dataToPlot[:, :, 0], origin='lower',vmin=-1, vmax=1)
    ani = animation.FuncAnimation(plt.gcf(),
                                AnimateFullStatus, fargs=(dataToPlot, timevec, img, ax1, probepositions, lineseriesdata, currentPlot, linedistance, probecolors,ylim),
                                frames=nframes, interval=50)

    return ani

def AnimateFullStatus(frameNR, fullstatus,timevec, img, ax1, probepositions, lineseriesdata, currentPlot, linedistance, probecolors, ylim):
    """
    Parameters
    ----------
    frameNR : int
    fullstatus : numpy.ndarray
    timevec : array-like
    img : matplotlib.image.AxesImage
    ax1 : matplotlib.axes.Axes
    probepositions : sequence of tuple of int
    lineseriesdata : numpy.ndarray
    currentPlot : matplotlib.axes.Axes
    linedistance : float
    probecolors : sequence
    ylim : array-like

    Returns
    -------
    None
    """
    # plt.figure(figsize=(10,10))
    img.set_data(fullstatus[ :, :, frameNR])
    #update time stamp in title
    ax1.set_title('Time =  ' + str(np.round(timevec[frameNR],3)))
  
    #  lineseriesdata[:][frameNR] = fullstatus[probepositions[:, 0], probepositions[:, 1], frameNR]
    # lineseriesdata[:][frameNR] += np.arange(len(probepositions)) * linedistance
    for ind, position in enumerate(probepositions):
        lineseriesdata[ind][frameNR] = fullstatus[position[0],position[1],frameNR]
        lineseriesdata[ind][frameNR] += ind * linedistance
    currentPlot.cla()
    currentPlot.set_ylim(ylim[0], len(probepositions * linedistance) +ylim[1])
    #currentPlot.set_yticks(np.arange(0,len(probepositions)*linedistance,linedistance),["O" for probe in probepositions])
    #ax1.tick_params(axis='y', colors=['red', 'black'], )  
    currentPlot.yaxis.set_visible(False)
    currentPlot.plot(lineseriesdata.T, linewidth =4)

    # Set x-ticks and x-tick labels at every 10th data point
    currentPlot.set_xticks(np.arange(0, len(timevec), 50))
    currentPlot.set_xticklabels(timevec[::50])
    for ind, line in enumerate(currentPlot.get_lines()):         
        line.set_color("black")
        #line.set_color(probecolors[ind])
        currentPlot.add_patch(plt.Rectangle((-2.5, (ind*linedistance)-0.25), 1, 0.5, facecolor='none',edgecolor=probecolors[ind],lw=8, clip_on=False))
    #currentPlot.get_lines()[3].set_color("red")

def plot_geodesic_distance_on_surface(vertices, faces,  path, chanInds, distance):
    """Plot a geodesic path and its endpoints on a triangular surface.

    Parameters
    ----------
    vertices : numpy.ndarray
        Three-dimensional surface vertices.
    faces : numpy.ndarray
        Triangular face indices.
    path : numpy.ndarray
        Three-dimensional coordinates along the geodesic path.
    chanInds : tuple of int
        Indices of the path start and end vertices.
    distance : float
        Geodesic distance displayed in the figure title.

    Returns
    -------
    None
        Displays an interactive Plotly figure.
    """
    # Create the 3D surface
    surface = go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        color='lightblue',
        opacity=0.5,
        name='Surface'
    )

    # Highlight the geodesic path
    path_coords = path  # Use path directly since it contains the coordinates
    geodesic_path = go.Scatter3d(
        x=path_coords[:, 0],
        y=path_coords[:, 1],
        z=path_coords[:, 2],
        mode='lines',
        line={"color": 'red', "width": 4},
        name='Geodesic Path'
    )

    # Highlight the start and end points
    start_end_points = go.Scatter3d(
        x=[vertices[chanInds[0], 0], vertices[chanInds[1], 0]],
        y=[vertices[chanInds[0], 1], vertices[chanInds[1], 1]],
        z=[vertices[chanInds[0], 2], vertices[chanInds[1], 2]],
        mode='markers+text',
        marker={"size": 8, "color": ['blue', 'green'], "symbol": 'circle'},
        text=[str(chanInds[0]), str(chanInds[1])],
        textposition='top center',
        name='Start/End Points'
    )

    # Highlight all vertex positions with indices
    vertices_plot = go.Scatter3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        mode='markers+text',
        marker={"size": 5, "color": 'black'},
        text=[str(i) for i in range(len(vertices))],
        textposition='top center',
        name='Vertices'
    )

    # Combine all plots
    plotData = [surface, vertices_plot, geodesic_path, start_end_points]

    # Layout
    layout = go.Layout(
        title=f'Geodesic Path on Surface (Distance: {distance:.2f})',
        scene={
            "xaxis_title": 'X',
            "yaxis_title": 'Y',
            "zaxis_title": 'Z',
        },
        showlegend=True,
        width=900,
        height=900
    )

    # Create the figure
    fig = go.Figure(data=plotData, layout=layout)
    fig.show()

def plot_topomap(waveData, dataBucketName=None, dataInds=None,timeInds= None, trlInd = None, type = None):
    """Interpolate channel values onto a two-dimensional topographic map.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object with channel data and two-dimensional coordinates.
    dataBucketName : str or None, default=None
        Name of the input data bucket. None uses the active bucket.
    dataInds : tuple or None, default=None
        Indices selecting data that reduce to trial, channel, and time.
    timeInds : tuple of int, int, or None, default=None
        Time interval to average, a single time index, or None to average all
        time points.
    trlInd : int or None, default=None
        Trial index to plot. None averages across trials.
    type : {"angle", "power", None}, default=None
        Transformation applied before interpolation.

    Returns
    -------
    None
        Displays a Matplotlib topographic map.
    """
    if dataBucketName is None:
        dataBucketName = waveData.ActiveDataBucket
    data = waveData.get_data(dataBucketName)[dataInds]
    if type == "angle":
        data = np.angle(data)
        plt.set_cmap('twilight')
    elif type == "power":
        data = np.abs(data)
    
    pos_2d = waveData.get_2d_coordinates()
    if timeInds is None: #average over time
        data = np.mean(data, axis=-1)
    elif isinstance(timeInds, tuple): #average between timepoints
        data = np.mean(data[:, :, timeInds[0]:timeInds[1]], axis=-1)
    elif isinstance(timeInds, int): #single timepoint
        data = data[:, :, timeInds]

    data = np.mean(data, axis=0) if trlInd is None else data[trlInd]
        
    # Create a grid to interpolate the data
    grid_x, grid_y = np.mgrid[
        pos_2d[:, 0].min():pos_2d[:, 0].max():100j,
        pos_2d[:, 1].min():pos_2d[:, 1].max():100j
    ]

    # Interpolate the data
    grid_z = griddata(pos_2d, data, (grid_x, grid_y), method='cubic')

    # Plot the topomap
    if type == "angle":
        img = plt.imshow(grid_z.T, extent=(pos_2d[:, 0].min(), pos_2d[:, 0].max(), pos_2d[:, 1].min(), pos_2d[:, 1].max()), origin='lower', vmin=-np.pi, vmax=np.pi)
    else:
        img = plt.imshow(grid_z.T, extent=(pos_2d[:, 0].min(), pos_2d[:, 0].max(), pos_2d[:, 1].min(), pos_2d[:, 1].max()), origin='lower')
    plt.colorbar(img)

def plot_optical_flow(waveData, PlottingDataBucketName = None, UVBucketName = None, dataInds = None,plotangle = False, normVectorLength=False):
    """Animate optical-flow vectors over the data used to compute them.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing the source data and complex optical-flow
        vectors.
    PlottingDataBucketName : str, default=None
        Name of the data bucket shown as the image background. This must be
        supplied.
    UVBucketName : str, default=None
        Name of the optical-flow bucket. By default, the active data bucket is
        used.
    dataInds : tuple, default=None
        Indices selecting a single spatial map over time from both buckets.
    plotangle : bool, default=False
        Plot phase angle with a cyclic colormap instead of the real component.
    normVectorLength : bool, default=False
        Normalize each optical-flow vector to unit magnitude before plotting.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Animation displaying background data with overlaid optical-flow
        vectors.
    """
    if UVBucketName is None:
        UVBucketName = waveData.ActiveDataBucket
    if PlottingDataBucketName is None:
        raise ValueError('Please specify a data bucket to plot')

    # Ensure consistency
    hf.assure_consistency(waveData)
    hf.assure_consistency(waveData, PlottingDataBucketName)

    # Get the data
    UV = np.squeeze(waveData.DataBuckets[UVBucketName].get_data()[dataInds])
    
    if normVectorLength:
        UV = UV/ np.abs(UV)
    if plotangle:
        plotData = np.squeeze(np.angle(waveData.get_data(PlottingDataBucketName))[dataInds])
        cmap = 'twilight'
    else:
        plotData = np.squeeze(np.real(waveData.get_data(PlottingDataBucketName))[dataInds])
        cmap = 'copper'

    nFrames = plotData.shape[-1]  # time is the last dimension
    timevec = waveData.get_time()

    def AnimateFullStatus(frameNR, fullstatus,timevec):
        img.set_data(fullstatus[ :, :, frameNR])
        #update time stamp in title
        ax1.set_title('Time =  ' + f"{timevec[frameNR]:.2f}")
        barbs.set_UVC(-np.real(UV[ :, :, frameNR]), -np.imag(
            UV[ :, :, frameNR]))

    fig = plt.figure(figsize=(7, 5))
    ax1 = plt.subplot()
    ax1.grid(None)
    vmin, vmax = np.percentile(plotData, [5, 95])
    img = ax1.imshow(plotData[:, :, 0], origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    barbs = ax1.quiver(-np.real(UV[:, :, 0]), -np.imag(UV[:, :, 0]))

    if plotangle:
        # Add a small subplot with the ring plot in the upper right corner
        ax2 = fig.add_axes([0.8, 0.7, 0.2, 0.2], polar=True)
        azimuths = np.radians(np.linspace(0, 360, 360))
        zeniths = np.linspace(0.4, 0.7, 30)
        r, theta = np.meshgrid(zeniths, azimuths)
        values = theta
        ax2.pcolormesh(theta, r, values, cmap='twilight')
        ax2.set_rgrids([0.4, 0.7], labels=[], angle=180)
        ax2.set_yticklabels([])
        ax2.grid(color='white')
        radian_multiples = [0, 0.5, 1, 1.5, 2]
        radians = [r * np.pi for r in radian_multiples]
        radian_labels = ['0', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', '']
        ax2.set_xticks(radians)
        ax2.set_xticklabels(radian_labels)
    else:
        cbar = plt.colorbar(img)
        cbar.set_label(r'$\mu$V')

    ani = animation.FuncAnimation(plt.gcf(),
                                AnimateFullStatus, fargs=(plotData, timevec),
                                frames=nFrames-1, interval=50)
    return ani

def plot_optical_flow_polar_scatter(waveData, UVBucketName=None, directionalStabilityBucket=None, dataInds=None, windowSize=100):
    """Animate optical-flow directions and directional stability in polar form.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing optical-flow and directional-stability data.
    UVBucketName : str or None, default=None
        Name of the complex optical-flow bucket.
    directionalStabilityBucket : str or None, default=None
        Name of the moving-window directional-stability bucket.
    dataInds : tuple or None, default=None
        Indices selecting one ``posx_posy_time`` data array.
    windowSize : int, default=100
        Number of time points displayed in each polar-scatter window.

    Returns
    -------
    matplotlib.animation.FuncAnimation
        Polar-scatter animation with directional-stability vectors.

    Raises
    ------
    ValueError
        If either required bucket name or ``dataInds`` is not supplied.
    """
   
    if UVBucketName is None:
        raise ValueError('Please specify a data bucket with UV information to plot')
    if directionalStabilityBucket is None:
        raise ValueError('Please specify a data bucket with the directional stability data')
    if dataInds is None:
        raise ValueError('Please specify the data indices to plot')

    # Get the data
    UV = waveData.DataBuckets[UVBucketName].get_data()[dataInds]
    averageVectors = waveData.DataBuckets[directionalStabilityBucket].get_data()[dataInds] 
    UnitVec = UV/ np.abs(UV)
    timevec = waveData.get_time()

    def AnimatePolarScatter(frameNr, UV, AverageVectors, WindowSize, timevec):
        currentUV = UV[:,:, frameNr:frameNr + WindowSize]
        offsetArray = np.stack((np.angle(currentUV).ravel(), np.abs(currentUV).ravel()), axis=1)
        scatterPlot.set_offsets(offsetArray)
        scatterPlot.set_sizes(offsetArray[:, 1] * 30)
        ax.set_title('Time ' + str(timevec[frameNr]))
        if(frameNr >= WindowSize):
            currentAverages = AverageVectors[:, :, (frameNr-WindowSize)+1].ravel()
            for idx, line in enumerate(lines):
                line.set_data([0, np.angle(currentAverages[idx])],
                            [0, np.abs(currentAverages[idx])])
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='polar')
    ax.set_ylim(0, 1.1)

    # Adjust the spacing to make room for the title
    plt.subplots_adjust(top=0.80)
    dimx, dimy, nFrames = UV.shape
    pad = np.zeros((dimx, dimy, windowSize))
    paddedUnitVec = np.concatenate((pad, UnitVec, pad), axis=-1)

    # Create a color map for the x-direction
    cmap_x = plt.cm.get_cmap('RdBu', dimx)
    colors_x = cmap_x(np.linspace(0, 1, dimx))

    # Create a color map for the y-direction
    cmap_y = plt.cm.get_cmap('PuOr', dimy)
    colors_y = cmap_y(np.linspace(0, 1, dimy))

    # Combine the color maps
    colors = np.zeros((dimx, dimy, 4))
    for i in range(dimx):
        for j in range(dimy):
            colors[i, j, :3] = (colors_x[i, :3] + colors_y[j, :3]) / 2  # Average the RGB values
            colors[i, j, 3] = (colors_x[i, 3] + colors_y[j, 3]) / 2  # Average the alpha values

    # Reshape the colors to a 1D array
    colors_1d = colors.reshape(-1, 4)

    # Repeat the colors for each time point
    allcolors = np.repeat(colors_1d, windowSize, axis=0)

    # Adjust the alpha values over time
    alphasteps = np.linspace(0.1, 1, windowSize)
    alphasteps = np.repeat(alphasteps, dimx * dimy)
    allcolors[:, 3] = alphasteps

    scatterPlot = ax.scatter(np.angle(pad),
                            np.abs(pad), s=20, color=allcolors)
    lines = ax.plot([np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                    np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])],
                    [np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1]),
                    np.zeros(paddedUnitVec.shape[0] * paddedUnitVec.shape[1])], marker='o', linewidth=1.5, markersize=8)
    for idx, line in enumerate(lines):
        line.set_color(colors_1d[idx])

    ani = animation.FuncAnimation(plt.gcf(),
                                AnimatePolarScatter, fargs=(
                                    paddedUnitVec, averageVectors, windowSize, timevec),
                                frames=nFrames-1, interval=100)
    return ani

def plot_streamlines(UV, seedpoints):
    """
    Parameters
    ----------
    UV : numpy.ndarray
    seedpoints : numpy.ndarray

    Returns
    -------
    pyvista.Plotter
    """
    #uv = np.dstack((np.zeros((UV.shape[0], UV.shape[1])), UV))
    nx = UV.shape[0]
    ny = UV.shape[1]
    nz = UV.shape[2]
    u = np.real(UV)
    v = np.imag(UV)
    # origin = (-(nx - 1) * 1 / 2, -(ny - 1) * 1 / 2, -(nz - 1) * 1 / 2) #Puts origin at centre
    origin = (0, 0, 0)
    mesh = pv.ImageData(dimensions=(nx, ny, nz), spacing=(1, 1, 1), origin=origin)
    vectors = np.zeros((u.shape[0] * u.shape[1], 3))
    for tt in range(UV.shape[2]):
        # Arrange 2d vector-fields in space-time(added 3rd dimension = time)
        newarray = np.stack(
            (np.ravel(u[:, :, tt]) ** 3, np.ravel(v[:, :, tt]) ** 3, np.ones(u[:, :, tt].size))).T
        vectors = newarray if tt == 0 else np.vstack((vectors, newarray))
    # Create polydata object
    mesh['vectors'] = vectors
    #create plotters
    pv.set_plot_theme("document")
    pv.vector_poly_data(mesh.points, vectors)
    sourcepoints = mesh.points[seedpoints.T.ravel()]
    wrappedPoints = pv.wrap(sourcepoints)
    stream = mesh.streamlines_from_source(
        wrappedPoints, 'vectors', integration_direction="forward",
        initial_step_length=0.5, max_step_length=0.5, min_step_length=0.5,
        interpolator_type="cell")

    # plotting vectors
    # pdata.glyph(orient='vectors', scale='mag').plot()
    # pdata.glyph(orient='vectors', scale=False).plot()
    #pl.add_mesh(pdata)
    #pl.show()
    # plot all streamlines
    p = pv.Plotter(off_screen=True)  # Note the off_screen argument
    tube = stream.tube(radius=0.05)
    p.add_mesh(tube)
    return p

def plot_polar_histogram(waveData, DataBucketName, dataInds=None):
    """Plot a magnitude-weighted polar histogram of directional stability.

    Parameters
    ----------
    waveData : WaveSpace.Utils.WaveData.WaveData
        WaveData object containing complex data. Should be the result of opticl flow analysis.
    DataBucketName : str
        Name of the data bucket. Should be the result of optical flow analysis.
    dataInds : tuple or None, default=None
        Indices of data to plot e.g.:(freqbin,trial).
        Dimensions after indexing should be posx_posy_time

    Returns
    -------
    matplotlib.figure.Figure
        Polar histogram with vector magnitudes used as weights.
    """
    waveData.set_active_dataBucket(DataBucketName)
    # Ensure consistency
    hf.assure_consistency(waveData)

    # Get the data
    Vectors = waveData.DataBuckets[DataBucketName].get_data()[dataInds]

    angles = np.angle(Vectors)
    magnitudes = np.abs(Vectors)
    angles = angles.ravel()
    magnitudes = magnitudes.ravel()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.hist(angles, bins=36, weights=magnitudes, color='b', alpha=0.7)
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)
    ax.set_yticklabels([])
    ax.set_facecolor('white')
    ax.grid(True, color='black')

    return fig


