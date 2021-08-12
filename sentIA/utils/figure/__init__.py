# Plotting tools and utility functions
# Nested GridSpec : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py
# GridSpec : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
# colorbar : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html#sphx-glr-gallery-subplots-axes-and-figures-colorbar-placement-py

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
from matplotlib import colors
import numpy as np
import os

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

class FigBase():
	"""
	"""

	CREDITS = "Credit : EU, contains modified Copernicus Sentinel data, processed with custom script."
	
	def __init__(self, title : str, dim : tuple):
		self.title = title
		self.fig = plt.figure(figsize=dim)

	def _format(self):
		pass

	def show(self):
		pass

###############
## Functions ##
###############

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)