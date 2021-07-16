# Masking processing functions

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

def water_mask(image, res : str, threshold : float = 0.2, reversed : bool = False):
	"""
	Return a mask based on NDWIwb index.
	"""
	ndwi = image.get_NDWIwb(res)
	mask = ndwi > threshold
	if reversed:
		return reverse(mask)
	else:
		return mask

def reverse(bool_array : np.ndarray):
	"""
	Return a reversed array.
	"""
	return ~bool_array

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)