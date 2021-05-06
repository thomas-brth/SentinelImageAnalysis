# Processing tools

#############
## Imports ##
#############

import numpy as np

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
	for i in range(3):
		res.append(crop_2D(image[:, :, i]))
	image = np.dstack(res)
	return image

def main():
	pass

if __name__ == '__main__':
	main()