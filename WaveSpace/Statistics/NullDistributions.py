#%%
import numpy as np
import WaveSpace.Utils.WaveData as wd
import WaveSpace.Utils.HelperFuns as hf

#%%
#mirrior-pad data array to all sides
def pad_data(data, padSize):
    """Reflect-pad the first two dimensions of a three-dimensional array.

    Parameters
    ----------
    data : numpy.ndarray
        Array with spatial dimensions first and time last.
    padSize : int
        Number of elements to add at each edge of both spatial dimensions.

    Returns
    -------
    numpy.ndarray
        Reflect-padded array.
    """
    data = np.pad(data, ((padSize, padSize), (padSize, padSize), (0, 0)), 'reflect')
    return data

def get_warp_field(gridSize, maxDistortion, nSteps):
    """Generate random displacement fields for a two-dimensional grid.

    Parameters
    ----------
    gridSize : tuple of int
        Number of grid points along the two spatial dimensions.
    maxDistortion : float
        Maximum displacement scale.
    nSteps : int
        Number of sequential distortion steps.

    Returns
    -------
    XIn, YIn : numpy.ndarray
        Horizontal and vertical displacement fields.

    Notes
    -----
    The fields are generated from random discrete-cosine components and scaled
    by ``maxDistortion / nSteps``.
    """

    gridX, gridY = gridSize
    ncomp = 6  # Number of components
    # Create a meshgrid
    YI, XI = np.meshgrid(np.arange(1, gridX+1), np.arange(1, gridY+1))

    # Initialize random phase and amplitude for DCTs
    ph = np.random.rand(ncomp, ncomp, 4) * 2 * np.pi
    a = np.random.rand(ncomp, ncomp) * 2 * np.pi

    # Initialize warp fields
    Xn = np.zeros((gridX, gridY))
    Yn = np.zeros((gridX, gridY))

    # Generate warp fields by adding random DCTs
    for xc in range(ncomp):
        for yc in range(ncomp):
            Xn += a[xc, yc] * np.cos(xc * XI / gridY * 2 * np.pi + ph[xc, yc, 0]) * np.cos(yc * YI / gridX * 2 * np.pi + ph[xc, yc, 1])
            Yn += a[xc, yc] * np.cos(xc * XI / gridY * 2 * np.pi + ph[xc, yc, 2]) * np.cos(yc * YI / gridX * 2 * np.pi + ph[xc, yc, 3])

    # Normalize to RMS of warps in each direction
    Xn = Xn / np.sqrt(np.mean(Xn**2))
    Yn = Yn / np.sqrt(np.mean(Yn**2))

    # Scale by maximum distortion and number of steps
    YIn = maxDistortion * Yn / nSteps
    XIn = maxDistortion * Xn / nSteps

    return XIn, YIn

def warp_array(data,maxDistortion, nSteps):
    """Apply one random warp field across the time axis of an array.

    Parameters
    ----------
    data : numpy.ndarray
        Three-dimensional array ordered as x, y, and time.
    maxDistortion : float
        Maximum displacement scale.
    nSteps : int
        Number of sequential distortion steps.

    Returns
    -------
    numpy.ndarray
        Reflect-padded and warped data array.

    Notes
    -----
    The same displacement field is used for every time point.
    """
    # pad array to avoid edge effects
    padSize = 10
    data = pad_data(data, padSize)
    arrayShape = data.shape
    nX, nY, nT = arrayShape
    # get warp fields
    XIn, YIn = get_warp_field((nX, nY), maxDistortion, nSteps)
    # copy data
    warpdata = data.copy()
    #loop over time and apply same warp field to each snapshot
    for t in range(nT):
        warpdata[:,:,t] = warp_snapshot(data[:,:,t], XIn, YIn)

    return warpdata



from scipy.ndimage import map_coordinates
from scipy.interpolate import interp2d
from skimage.color import rgb2gray
from skimage.transform import resize

def warp_snapshot(data, XIn, YIn, phaseoffset=40):
    """Warp spatial data with supplied displacement fields.

    Parameters
    ----------
    data : numpy.ndarray
        Array whose first two dimensions are spatial and whose final dimension
        is iterated during interpolation.
    XIn, YIn : numpy.ndarray
        Horizontal and vertical displacement fields.
    phaseoffset : int, default=40
        Retained for compatibility; it is not used by the implementation.

    Returns
    -------
    numpy.ndarray
        Resized grayscale warped image.

    Notes
    -----
    The displacement fields are applied in four quadrants before converting
    the interpolated result to grayscale.
    """
    imsz = data.shape[0]
    YI, XI = np.mgrid[0:imsz, 0:imsz]

    interpIm = data.copy()

    for quadrant in range(1, 5):
        if quadrant == 1:
            cx, cy = XIn, YIn
            ind = 1
        elif quadrant == 2:
            cx, cy = XIn - XIn, YIn - YIn
        elif quadrant == 3:
            ind = 4
            interpIm = data.copy()
            cx, cy = XIn, YIn
        elif quadrant == 4:
            cx, cy = XIn - XIn, YIn - YIn

        cy = YI + cy
        cx = XI + cx
        mask = (cx < 1) | (cx > imsz) | (cy < 1) | (cy > imsz)
        cx[mask] = 1
        cy[mask] = 1

        for i in range(interpIm.shape[2]):
            interpIm[:,:,i] = interp2d(np.arange(imsz), np.arange(imsz), interpIm[:,:,i])(cy, cx)

    diffIm = resize(rgb2gray(interpIm), (imsz//2, imsz//2))

    return diffIm


