# General utility functions

#############
## Imports ##
#############

import enum

###############
## Constants ##
###############

#############
## Classes ##
#############

class SCL_Labels(enum.Enum):
	"""
	Scene classification labels, as processed by sen2cor algorithm.
	"""
	NO_DATA = 0
	SATURATED_OR_DEFECTIVE = 1
	DARK_AREA_PIXELS = 2
	CLOUD_SHADOWS = 3
	VEGETATION = 4
	NOT_VEGETATED = 5
	WATER = 6
	UNCLASSIFIED = 7
	CLOUD_MEDIUM_PROBABILITY = 8
	CLOUD_HIGH_PROBABILITY = 9
	THIN_CIRRUS = 10
	SNOW = 11

class Band(enum.Enum):
	"""
	Sentinel-2 band enumeration.
	It gives the name of each band, the resolutions for which it is available and its central wavelength in nm.
	"""
	B01 = ('Coastal aerosol', ['R60m'], 443)
	B02 = ('Blue', ['R10m', 'R20m', 'R60m'], 490)
	B03 = ('Green', ['R10m', 'R20m', 'R60m'], 560)
	B04 = ('Red', ['R10m', 'R20m', 'R60m'], 665)
	B05 = ('Re1', ['R20m', 'R60m'], 705)
	B06 = ('Re2', ['R20m', 'R60m'], 740)
	B07 = ('Re3', ['R20m', 'R60m'], 783)
	B08 = ('NIR', ['R10m'], 842)
	B8A = ('NIRn', ['R20m', 'R60m'], 865)
	B09 = ('Water vapor', ['R60m'], 940)
	B11 = ('SWIR1', ['R20m', 'R60m'], 1610)
	B12 = ('SWIR2', ['R20m', 'R60m'], 2190)
	SCL = ('Scene classification', ['R20m', 'R60m'], None)

	def __init__(self, band_name : str, resolutions : list, wavelength : int):
		self.band_name = band_name
		self.resolutions = resolutions
		self.wavelength = wavelength

	def __str__(self):
		"""
		"""
		return f"{self.name} : {self.band_name} - {self.wavelength}nm ({', '.join(self.resolutions)})"

	@classmethod
	def get_bands(cls, res : str):
		"""
		Return all bands available for a given resolution.
		"""
		return [band for band in cls if band.is_available(res)]

	def is_available(self, res : str):
		"""
		Return True if the band is available for a given resolution.
		"""
		return res in self.resolutions

###############
## Functions ##
###############

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)