#Star Photometry Curve fitting
#https://stackoverflow.com/questions/21566379/fitting-a-2d-gaussian-function-using-scipy-optimize-curve-fit-valueerror-and-m
"""
STAR PHOTOMETRY — DARK SUBTRACTION + 2D GAUSSIAN FIT
Smear in long exposures means point sources are no longer circular.
This script fits a *tilted elliptical 2D Gaussian*, which handles smear.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from astropy.io import fits

##NOTICE, in  current version, the images must be converted from compressed SBIG to FITS through CCDOPS. 
# ------------------------------------------------------
# 1. LOAD SCIENCE + DARK & DARK-SUBTRACT
# ------------------------------------------------------

science_file = r"C:\Users\tessa\Documents\CCD2_20250204_20s.SBIG"
dark_file    = r"C:\Users\tessa\Documents\CCD2_20250204_20s_dark1.SBIG"

def load_fits(path):
    """Loads a FITS file and returns (data, header)."""
    h = fits.open(path)                         #opens SBIG image files
    arr = h[0].data.astype(float)               #the image array (16-bit CCD counts)
    hdr = h[0].header                           #the FITS header for the primary image
    h.close()
    #print(hdr)
    return arr, hdr

science, hdr_sci = load_fits(science_file)  
dark, hdr_dark   = load_fits(dark_file)

def get_exptime(header):
    """Returns exposure time from a FITS header (best guess)."""
    for key in ("EXPTIME", "EXPOSURE", "EXPTM", "EXPT"):
        if key in header:
            try:
                return float(header[key])
            except:
                pass
    return 1.0

exp_sci = get_exptime(hdr_sci)                  #gets the exposure time from the SBIG header
exp_dark = get_exptime(hdr_dark)                #gets the exposure time from the SBIG header

# scale dark to match science exposure
# A dark frame collects dark current, which is causes by thermal electrons accumulating during exposure. 
dark_scaled = dark * (exp_sci / exp_dark)       #scales the dark frame if it is not the same exposure time as the image capture

# subtract darkframe from original image. 
data = science - dark_scaled


# ------------------------------------------------------
# 2. ASINH STRETCH FOR DISPLAY
# ------------------------------------------------------

#Removes the background, measures noise level, uses the arcsing function to strech faint pizels while protecting bright ones, produces a nice image where both bright and faint stars are visible
#This function applies a nonlinear brightness stretch commonly used in astronomy to make faint stars visible without blowing out the bright ones
def asinh_stretch(img, scale=0.1):
    """Nonlinear stretch for astronomy images."""
    #centers the image around zero
    #Astonomy images have a "background level" (sky glow, read noise). Subtracting the median removes most of that, so:
        #Background becomes zero, stars and objects stand out above zero.
    img = img - np.median(img)
    #scale*np.std(img) is the standard deviation of the image (the measure of how noisy or spread out the pixel values are)
    #the scale is a small constant to contol contrast  
    #np.arcsinh(): arcsinh is applied because it behaves like linear for small values (so faint stars become visible), and logarithmic for large values (so bright stars don't saturate)
    return np.arcsinh(img / (scale * np.std(img)))

#displays a CCD image using percentile-based contrast scaling, which makes the image easier to see wihtout being washed out by very bright or very dark pixels. 
def show_image(img, title=""):
    #finds the intensity values at the 1st percentile (very faint pixels) and the 99th percentile (very bright pixels)
    #this is done becaues there can be hot pixels, saturated stars, and noisy background, which distort the constrast
    #percentile cliping ignores the outliers and gives a clear display range, this makes faint stars visible without blowing out the bright ones
    lo, hi = np.percentile(img, (1, 99))
    plt.figure(figsize=(6,6))
    #origin='lower': makes pixel (0,0) start at the bottom-left (so image matches sky coordinates)
    #vmin=lo, vmax=hi: sets brightness range from the 1st-99th range
    #imshow() treats a 2D array as an image. Maps data values as colors, uses pixel coordinates as axes, lets you set brightness limits
    plt.imshow(img, origin='lower', cmap='gray', vmin=lo, vmax=hi)
    plt.title(title)
    plt.show()


show_image(data, "Dark-Subtracted Frame")


# ------------------------------------------------------
# 3. INTERACTIVE STAR SELECTION
# ------------------------------------------------------

#plt.figure(): creates a new figure window for plotting. Can add multiple pots using plt.plot(), plt.imshow()
#it relies on Matplotlib's state-matchine interface (pyplot). You don't get direct access to the the figure or axes objects unless you explicitly call fig = plt.gcf() or ax = plt.gca()

#fig, ax = plt.subplots()
#Creates a figure and one or more axes objects at the same time. 
#figure vs. Axes
#figure (fig) = the entire window or page that holds everything
#axes (Ax) =  the actual plotting area ( the x/y plot where your image or line appears)
#advantages are: explicit control over axes and figure, and it is easier to make multiple subplots


fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(asinh_stretch(data), cmap='gray', origin='lower')
ax.set_title("Zoom/pan this window to find star, then close tab")
plt.show()
new_x = ax.get_xlim()
new_y = ax.get_ylim()
print("Saved xlim: ", new_x)
print("Saved ylim: ", new_y)

#apply the zoomed/panned view
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.imshow(asinh_stretch(data), cmap='gray', origin='lower')
ax2.set_title("Click on the star now")
ax2.set_xlim(new_x)
ax2.set_ylim(new_y)

#capture click
coords = plt.ginput(1, timeout=-1)
if not coords:
    raise SystemExit("No star selected.")

plt.close(fig2)
#map(): applies another function to each item in an iterable
#map() takes a function (int) and an interable(coords[0]), so it applies int(951.23) -> 951 and int(1013.77) -> 1013
x0, y0 = map(int, coords[0])
print(f"Selected pixel: x={x0}, y={y0}")

# ------------------------------------------------------
# 4. CUTOUT AROUND THE STAR
# ------------------------------------------------------

cutout_size = 20
ymin, ymax = y0 - cutout_size, y0 + cutout_size
xmin, xmax = x0 - cutout_size, x0 + cutout_size

# Safety: avoid edge crashes
if ymin < 0 or xmin < 0 or ymax > data.shape[0] or xmax > data.shape[1]:
    raise ValueError("Cutout would go outside the image. Choose a star farther from edge.")

star_cutout = data[ymin:ymax, xmin:xmax]

plt.figure(figsize=(6,6))
plt.imshow(asinh_stretch(star_cutout), cmap='gray', origin='lower')
plt.title("Zoomed-in Star Cutout")
plt.show()

# Coordinates within cutout
#y, a 2D array the size of star cutout where each row has the same y-coordinate
#x, a 2D array the size of star_cutout where each row has the same x-coordinate
#together they define the 2D coordinate for every pixel
#to evaluate a Gaussian at every picel, you need to know each pixel's x coordinate and each pixels y coordinate
y, x = np.indices(star_cutout.shape)


# ------------------------------------------------------
# 5. SMEAR-AWARE 2D GAUSSIAN (ELLIPTICAL + ROTATION)
# ------------------------------------------------------

#returns a 1D array because it matches the flattened shape used during fitting
def Gaussian2D(xy, A, x0, y0, sigx, sigy, theta, offset):
    """
    Elliptical, rotated 2D Gaussian.
    This handles smear/elongation because sigx != sigy and theta != 0.
    """
    x, y = xy
    x0, y0 = float(x0), float(y0)

    a = (np.cos(theta)**2)/(2*sigx**2) + (np.sin(theta)**2)/(2*sigy**2)
    b = -(np.sin(2*theta))/(4*sigx**2) + (np.sin(2*theta))/(4*sigy**2)
    c = (np.sin(theta)**2)/(2*sigx**2) + (np.cos(theta)**2)/(2*sigy**2)

    return (offset + A*np.exp(-(a*(x-x0)**2 + 2*b*(x-x0)*(y-y0) + c*(y-y0)**2))).ravel()


# ------------------------------------------------------
# 5. INITIAL GUESSES (MUCH BETTER)
# ------------------------------------------------------


#np.max(star_cutout) is the brightest pixel in the star -> approximate peak
#np.median(star_cutout) is the background sky brightness
#the ampltiude of the Gaussian is the peak brightness - the background brightness
A_guess = np.max(star_cutout) - np.median(star_cutout)
#assumes that the star is in the center of the selection by the user
x0_guess = star_cutout.shape[1] / 2
y0_guess = star_cutout.shape[0] / 2
#sigma width guess. A star typically spreads over a few pixels depending on seeing conditions
#sigma = 3px corresponds to the FWHM of around 7 pixels (since FWHM = 2.355*sigma)
#even if the smear enlongates the star, this is still close enough that curve_fit can refine it
sigx_guess = sigy_guess = 3.0
#theta is the rotation angle of the enlongated Gaussian. Starting at 0 means "we don't know the orientation, but the algorithm will figure it out". Even if it is wrong, cure_fit converges well
theta_guess = 0
#estimated sky background level. Using the median avoids being affected by bright star pixels. The background is usually faily constant across the small cutout
offset_guess = np.median(star_cutout)

initial_guess = (
    A_guess,
    x0_guess,
    y0_guess,
    sigx_guess,
    sigy_guess,
    theta_guess,
    offset_guess
)

# Increase maxfev so solver has more tries
#popt = "optimized parameters", this is the tuple that best fits the star
#pcov = covariance matrix. Used to estimate the uncertainties on the fitted parameters.
popt, pcov = curve_fit(     #curve fit tries to find the best parameters of the model function so it matches the data. 
    Gaussian2D,             #model function Gaussian2D
    (x, y),                 #x and y coordinates of every pixel in the star cutout
    star_cutout.ravel(),    #star_cutout is a 2D image, .ravel() flattens it into 1D array because curve_fit expects a 1D list of values. This is the pixel intensity data that the model must match
    p0=initial_guess,       #the inital guess is a tuple
    maxfev=20000            #it lets the solver try 20000 times before giving ip
)


#popt is the array for best-fit paramets returned by curve_fit
#if popt is [A, x0, y0, sigma_x, sigma_y, theta, offset]
#popt* unpacks them into the function as Gaussian2D((x, y), A, x0, y0, sigx, sigy, theta, offset)
#This call computes the Gaussian model value at every pixel coordinate (x, y)
#Because the Gaussian model is calculated as  aflat 1D vector, we need to turn it back into a 2D image
#This is done to visualize how well the fit matches the star. The original cutout: real intensities
#data_fitted: the smooth Gaussian model
data_fitted = Gaussian2D((x, y), *popt).reshape(star_cutout.shape)

plt.figure()
plt.imshow(star_cutout, cmap='gray', origin='lower')
#draws contour lines (level lines) on top of the image showing the shape of the fitted Gaussian
#A contour line is a cruve connecting points of equal value. Contour shows "constant intensity"
plt.contour(data_fitted, colors='r', linewidths=1)
plt.title("Fitted Gaussian (Red) Over Star")
plt.show()

#Peak intensity

A, x0, y0, sigx, sigy, theta, offset = popt
peak_intensity = A + offset
print("Peak intensity: ", (A + offset))

#Total Flux from Gaussian Equation
#Only uses the fitted parameters: amplitude (A) and standard deviations (sigx, sigy)
#Ignores pixel discretization and small scale noise
#Gives a smooth, analytic estimate of the star's total flux
#Best for when you trust the Gaissian fit; less sensitive to noise
total_flux = 2*np.pi*A*sigx*sigy
print("Total flux from Gaussian:", total_flux)

#Background-crrected aperature flux
#gives total brightness above the background floor
#Directly sums the pixel intensities inside the cutout. SUbtracts the background offset. Sensitive to pixel noise, nearby stars, imperfect centering. Closer to what real aperature does in astronomy
flux = np.sum(star_cutout - offset)
print("Background corrected aperature flux: ", flux)

#Flux from fitted image
#Sums the Gaussian model image (reshaped back into 2D)
#compbines the advantages of the first two methods: smooth model, less noisy than raw pixel sum
#still computed over the discrete pixel grid, so matches the data
#the "model-integrated" flux
data_fitted = np.sum(data_fitted - offset)
print("Flux from fitted image: ", data_fitted)


#Analytic Gaussian is used for total flux if the fit is good
#Aperature flux is used for quick checks, or if the star is not perfectly Gaussian
#Fitted image flux is an in-between, often ued to compare model vs. data

#When all three are close in value, it means the Gaussian fit is good, the background subtraction is accurate, and the noise is low/ star is well isolated


