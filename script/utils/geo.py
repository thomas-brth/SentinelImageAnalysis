# Geographic manipulation
# LINK: https://stackoverflow.com/questions/40342355/how-can-i-generate-a-regular-geographic-grid-using-python

#############
## Imports ##
#############

from shapely import geometry, wkt, ops
import pyproj
from functools import partial

###############
## Constants ##
###############

#############
## Classes ##
#############

###############
## Functions ##
###############

def geometry_from_wkt(geom : str):
	"""
	Parse the given geometry string and return a shapely object.
	"""
	return wkt.loads(geom)

def get_regular_grid(footprint : str, grid_step : int):
	"""
	"""
	# Projection transformers, EPSG:3857 is the metric crs
	to_proxy_transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')
	to_original_transformer = pyproj.Transformer.from_crs('epsg:4326', 'epsg:3857')

	# Extract image bounding box
	polygon = geometry_from_wkt(geom=footprint)
	llcrnrlon, llcrnrlat, urcrnrlon, urcrnrlat = polygon.bounds

	# Transform bounding box corners
	transformed_ll = to_proxy_transformer.transform(llcrnrlon, llcrnrlat)
	transformed_ur = to_proxy_transformer.transform(urcrnrlon, urcrnrlat)

	lons = []
	lats = []
	x = transformed_ll[0]
	while x < transformed_ur[0]:
		y = transformed_ll[1]
		while y < transformed_ur[1]:
			p = geometry.Point(to_original_transformer.transform(x, y))
			lons.append(p.x)
			lats.append(p.y)
			y += grid_step
		x += grid_step
	return lons, lats

def get_area(geom : str):
	"""
	Compute the area (in square kilometers) of a given footprint.
	"""
	polygon = geometry_from_wkt(geom)
	geod = pyproj.Geod(ellps='WGS84')
	return abs(geod.geometry_area_perimeter(polygon)[0]) / 1000000

def main():
	footprint = 'POLYGON((6.8338 50.4100,6.9646 50.4100,6.9646 50.4791,6.8338 50.4791,6.8338 50.4100))'
	area = get_area(footprint)
	print(area)

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)