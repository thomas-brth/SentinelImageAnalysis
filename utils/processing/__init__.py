# Processing tools

#############
## Imports ##
#############

## General imports
import numpy as np
from scipy import interpolate
from skimage import morphology
import cv2 as cv
from matplotlib import colors

## Custom imports
from mask import water_mask

###############
## Constants ##
###############

#############
## Classes ##
#############

class MidpointNormalize(colors.Normalize):
	"""
	Useful object enbling to normalize colorbar with a chosen midpoint.
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)
	
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0,0.5,1]
		return np.ma.masked_array(np.interp(value, x, y))

###############
## Functions ##
###############

def crop_2D(arr : np.array):
	"""
	Crop a 2D array by removing all "0" on the outer boundary.
	"""
	n, m = arr.shape
	arr_mask = morphology.binary_opening(arr == 0)
	while arr_mask[0, :].any() or arr_mask[:, 0].any() or arr_mask[n-1, :].any() or arr_mask[:, m-1].any():
		if arr_mask[0, :].any():
			arr = arr[1:, :]
			arr_mask = arr_mask[1:, :]
			n -= 1
		if arr_mask[:, 0].any():
			arr = arr[:, 1:]
			arr_mask = arr_mask[:, 1:]
			m -= 1
		if arr_mask[n-1, :].any():
			arr = arr[:n-1, :]
			arr_mask = arr_mask[:n-1, :]
			n -= 1
		if arr_mask[:, m-1].any():
			arr = arr[:, :m-1]
			arr_mask = arr_mask[:, :m-1]
			m -= 1
	return arr

def crop_3D(image : np.array):
	"""
	Apply the crop_2D function on each image band.
	This function enables to adjust the returned satellite imagery.
	"""
	res = []
	for i in range(3):
		res.append(crop_2D(image[:, :, i]))
	image = np.dstack(res)
	return image

def interpolation_2d(arr : np.ndarray, res_sup : int):
	"""
	Perform a bicubic interpolation over a given array.
	"""
	n, m = arr.shape
	x, y = np.arange(0, n, 1), np.arange(0, m, 1)
	f = interpolate.interp2d(y, x, arr, kind='cubic')
	xx, yy = np.arange(0, n, 1 / res_sup), np.arange(0, m, 1 / res_sup)
	return f(yy, xx)

def sharpen(arr : np.ndarray):
	"""
	Sharpen a given 2D array.
	"""
	kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
	return cv.filter2D(arr, -1, kernel)

def water_spectral_analysis(image, threshold : float = 0.2, reversed : bool = False):
	"""
	Compute for each band the average value and the standard deviation of every water pixels in a given image.
	"""
	bands = {
		'B01' : {'res' : 'R60m', 'wl': 443, 'mean' : 0, 'std' : 0},
		'B02' : {'res' : 'R10m', 'wl': 490, 'mean' : 0, 'std' : 0},
		'B03' : {'res' : 'R10m', 'wl': 560, 'mean' : 0, 'std' : 0},
		'B04' : {'res' : 'R10m', 'wl': 665, 'mean' : 0, 'std' : 0},
		'B05' : {'res' : 'R20m', 'wl': 705, 'mean' : 0, 'std' : 0},
		'B06' : {'res' : 'R20m', 'wl': 740, 'mean' : 0, 'std' : 0},
		'B07' : {'res' : 'R20m', 'wl': 783, 'mean' : 0, 'std' : 0},
		'B08' : {'res' : 'R10m', 'wl': 842, 'mean' : 0, 'std' : 0},
		'B8A' : {'res' : 'R20m', 'wl': 865, 'mean' : 0, 'std' : 0},
		'B09' : {'res' : 'R60m', 'wl': 940, 'mean' : 0, 'std' : 0},
		'B11' : {'res' : 'R20m', 'wl': 1610, 'mean' : 0, 'std' : 0},
		'B12' : {'res' : 'R20m', 'wl': 2190, 'mean' : 0, 'std' : 0},
	}
	for band in bands.keys():
		mask = water_mask(image=image, res=bands[band]['res'], threshold=threshold, reversed=reversed)
		band_arr = image.load_single_band(res=bands[band]['res'], band=band, factor=1/10000)
		band_arr[band_arr > 1] = 1
		bands[band]['mean'] = np.mean(band_arr[mask])
		bands[band]['std'] = np.std(band_arr[mask])
	return bands

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)