# Plotting tools and utility functions
# Nested GridSpec : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_nested.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-nested-py
# GridSpec : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/gridspec_multicolumn.html#sphx-glr-gallery-subplots-axes-and-figures-gridspec-multicolumn-py
# colorbar : https://matplotlib.org/stable/gallery/subplots_axes_and_figures/colorbar_placement.html#sphx-glr-gallery-subplots-axes-and-figures-colorbar-placement-py
# Tick params : https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.tick_params.html

# FigBase --> FigComp
#		  --> FigUnique
# 		  --> FigAnim

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import numpy as np
import json
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
	Parent class for all Figures.
	"""

	CREDITS = "Credit : EU, contains modified Copernicus Sentinel data, processed with custom script."
	
	def __init__(self, suptitle : str, dim : tuple):
		self.suptitle = suptitle
		self.fig = plt.figure(figsize=dim, constrained_layout=True)
		self.fig_ss = self._load_stylesheet()

	@classmethod
	def _load_stylesheet(cls):
		"""
		Load specific stylesheet (JSON file), to extract some parameters (colors, font, font color, font size...)
		"""
		with open(os.path.join(os.path.join(os.path.dirname(__file__), "resources"), "stylesheet.json"), 'r') as foo:
			d = json.load(foo)
		return d

	@property
	def axes(self):
		"""
		Return figure axes as a list.
		"""
		return self.fig.axes

	@property
	def naxes(self):
		"""
		Return number of figure axes.
		"""
		return len(self.axes)

	def function():
		pass

	def _format(self, naxes : int):
		"""
		Format figure axes.
		To be defined for each child (depends on the chosen gridspec)
		"""
		raise NotImplementedError

	def add_image(self, title : str, data : np.ndarray):
		"""
		"""
		raise NotImplementedError

	def add_index(self, title : str, data : np.ndarray, cmap : str, norm : colors.Normalize, cb_orientation : str):
		"""
		"""
		raise NotImplementedError

	def display(self):
		"""
		Display figure.
		To be defined for each child (depends on the display mode - Mere figure or animation).
		"""
		raise NotImplementedError

class FigStatic(FigBase):
	"""
	"""
	def __init__(self, suptitle : str, dim : tuple, naxes : int):
		super(FigStatic, self).__init__(suptitle=suptitle, dim=dim)
		self._format(naxes=naxes)
		self._gen_axes = (ax for ax in self.axes)

	def _format(self, naxes : int):
		"""
		"""
		## Axes creation (Gridspec, depends on the number of axes)
		if naxes == 1:
			gs = GridSpec(1, 1, figure=self.fig)
			self.fig.add_subplot(gs[0, 0])
		elif naxes == 2:
			gs = GridSpec(1, 2, figure=self.fig)
			self.fig.add_subplot(gs[0, 0])
			self.fig.add_subplot(gs[0, 1])
		elif naxes == 3:
			gs = GridSpec(2, 2, figure=self.fig)
			self.fig.add_subplot(gs[0, 0])
			self.fig.add_subplot(gs[0, 1])
			self.fig.add_subplot(gs[1, :])
		elif naxes == 4:
			gs = GridSpec(2, 2, figure=self.fig)
			self.fig.add_subplot(gs[0, 0])
			self.fig.add_subplot(gs[0, 1])
			self.fig.add_subplot(gs[1, 0])
			self.fig.add_subplot(gs[1, 1])
		else:
			raise Exception()
			
		## Color settings
		plt.rcParams['axes.facecolor'] = self.fig_ss['facecolor']
		self.fig.patch.set_facecolor(self.fig_ss['facecolor'])
		self.fig.suptitle(self.suptitle, **self.fig_ss['font']['suptitle'])

		for ax in self.axes:
			ax.spines['top'].set_color(self.fig_ss['contour_color'])
			ax.spines['left'].set_color(self.fig_ss['contour_color'])
			ax.spines['bottom'].set_color(self.fig_ss['contour_color'])
			ax.spines['right'].set_color(self.fig_ss['contour_color'])
			ax.tick_params(**self.fig_ss['tick_params'])
			ax.set_facecolor(self.fig_ss['facecolor'])

	def add_image(self, title : str, data : np.ndarray):
		"""
		"""
		ax = next(self._gen_axes)
		ax.imshow(data)
		ax.set_title(title, **self.fig_ss['font']['title'])

	def add_index(self, title : str, data : np.ndarray, cmap : str, norm : colors.Normalize, cb_orientation : str):
		"""
		"""
		ax = next(self._gen_axes)
		self.fig.add_axes(ax)

		im = ax.imshow(data, cmap=cmap, norm=norm)
		ax.set_title(title, **self.fig_ss['font']['title'])
		col = self.fig.colorbar(im, ax=ax, **self.fig_ss['colorbar'][cb_orientation])
		col.outline.set_edgecolor(self.fig_ss['colorbar']['outline'])

	def display(self):
		"""
		"""
		plt.show()

###############
## Functions ##
###############

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)