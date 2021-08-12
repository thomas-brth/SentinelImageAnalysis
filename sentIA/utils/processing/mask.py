# Masking processing functions

#############
## Imports ##
#############

import numpy as np
from skimage import morphology

###############
## Constants ##
###############

#############
## Classes ##
#############

###############
## Functions ##
###############

def water_mask(image, res : str, threshold : float = 0.2, reversed : bool = False):
	"""
	Return a mask based on NDMI index.
	"""
	ndmi = image.get_NDWI(res=res)
	mask = ndmi > threshold
	if reversed:
		return reverse(mask)
	else:
		return mask

def scl_based_mask(image, res : str, reversed : bool = False):
	"""
	Return a mask based on SCL values (cloud, snow and water mask).
	"""
	scl = image.get_SCL(res=res)
	mask = morphology.binary_opening((scl == 6) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11))
	if reversed:
		return reverse(mask)
	else:
		return mask

def reverse(bool_array : np.ndarray):
	"""
	Reverse a boolean mask.
	"""
	return ~bool_array

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)