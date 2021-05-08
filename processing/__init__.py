# Processing tools

#############
## Imports ##
#############

## General imports
import numpy as np

## Custom imports
from mask import water_mask

###############
## Constants ##
###############

#############
## Classes ##
#############

###############
## Functions ##
###############

def crop_2D(arr : np.array):
	n, m = arr.shape
	while np.isnan(arr[0, :]).any() or np.isnan(arr[:, 0]).any() or np.isnan(arr[n-1, :]).any() or np.isnan(arr[:, m-1]).any():
		if np.isnan(arr[0, :]).any():
			arr = arr[1:, :]
			n -= 1
		if np.isnan(arr[:, 0]).any():
			arr = arr[:, 1:]
			m -= 1
		if np.isnan(arr[n-1, :]).any():
			arr = arr[:n-1, :]
			n -= 1
		if np.isnan(arr[:, m-1]).any():
			arr = arr[:, :m-1]
			m -= 1
	return arr

def crop_3D(image : np.array):
	res = []
	# Image pre-treatment : previously, non-finite values have been replaced by 0.
	# It is required to replace them back with np.nan, so that we can use  crop_2D function.
	null_mask = np.dstack([image[:, :, 0] == 0, image[:, :, 1] == 0, image[:, :, 2] == 0])
	image[null_mask] = np.nan
	for i in range(3):
		res.append(crop_2D(image[:, :, i]))
	image = np.dstack(res)
	return image

def water_spectral_analysis(image, threshold : float = 0.2, reversed : bool = False):
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
		band_arr = image.load_single_band(res=bands[band]['res'], band=band) / 10000
		band_arr[band_arr > 1] = 1
		bands[band]['mean'] = np.mean(band_arr[mask])
		bands[band]['std'] = np.std(band_arr[mask])
	return bands


def main():
	pass

if __name__ == '__main__':
	main()