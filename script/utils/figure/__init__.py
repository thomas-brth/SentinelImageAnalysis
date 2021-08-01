# Plotting tools and utility functions

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np

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
		super(MidpointNormalize, self).__init__(vmin, vmax, clip)
		self.midpoint = midpoint
	
	def __call__(self, value, clip=None):
		x, y = [self.vmin, self.midpoint, self.vmax], [0,0.5,1]
		return np.ma.masked_array(np.interp(value, x, y))

###############
## Functions ##
###############

def main():
	pass

if __name__ == '__main__':
	main()