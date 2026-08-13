import numpy as np
from scipy.interpolate import interp2d
from skimage.color import rgb2gray
from skimage.transform import resize


#mirrior-pad data array to all sides
def pad_data(data, padSize):
    """
    Mirrior-pad the first two dimensions of the 3 dimensinal data array to all sides.

    Parameters:
    data (np.array): Data array.
    padSize (int): Size of padding.

    Returns:
    data (np.array): Padded data array.
    """
    data = np.pad(data, ((padSize, padSize), (padSize, padSize), (0, 0)), 'reflect')
    return data

def get_warp_field(gridSize, maxDistortion, nSteps):
    """
    Generates a diffeomorphic warp field by adding random discrete cosine transforms.
    Based on Stojanoski, B., & Cusack, R. (2014). Time to wave good-bye to phase scrambling: 
    Creating controlled scrambled images using diffeomorphic transformations. 
    Journal of Vision, 14(12), 6. doi:10.1167/14.12.6
    
    Parameters:
    gridSize (int,int): size of grid.
    maxDistortion (float): Maximum distortion.
    nSteps (int): Number of steps.

    Returns:
    XIn, YIn (np.array): Diffeomorphic warp fields.
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
    """ Apply same diffeomorphic transformation to each snapshot of data array (2 spatial dims)
    Parameters:
    data (np.array): Data array. shape(nX,nY,nT)
    maxDistortion (float): Maximum distortion. 
    nSteps (int): Number of steps.
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


def warp_snapshot(data, XIn, YIn, phaseoffset=40):
    """
    Apply diffeomorphic transformation to snapshot of data array (2 spatial dims)
[cxA cyA]=getdiffeo(imsz,distortion);
[cxB cyB]=getdiffeo(imsz,distortion);
[cxF cyF]=getdiffeo(imsz,distortion);
    Parameters:
    data (np.array): Data array. shape(nX,nY)
    cxA, cyA, cxB, cyB, cxF, cyF (np.array): Diffeomorphic warp fields.
    phaseoffset (int): Phase offset for the transformation.
    """
    imsz = data.shape[0]
    YI, XI = np.mgrid[0:imsz, 0:imsz]

    interpIm = data.copy()

    for quadrant in range(1, 5):
        if quadrant == 1:
            cx, cy = XIn, YIn
        elif quadrant == 2:
            cx, cy = XIn - XIn, YIn - YIn
        elif quadrant == 3:
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


