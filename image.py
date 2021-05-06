# Sentinel-2 satellite data extraction and processing script

#############
## Imports ##
#############

## Sentinelsat imports ##
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt #create and retrieve a query

## General imports ##
import os
import json # for metadata
import zipfile # unzip collected data
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

## Rasterio imoprts ##
import rasterio as rio # open and read images (.jp2, .tiff, .tif...)
from rasterio import plot # used mainly for reshaphe_as_image() and reshape_as_raster() functions
from rasterio import mask as msk # used to crop images

## Datetime imports ##
from datetime import date
from datetime import timedelta as td

## Geometry imports ##
import fiona # to manage GeoJsons 
from fiona import transform as tf # for crs transformation

## Matplotlib ##
from matplotlib import pyplot as plt # display images

## Image Saving ##
import cv2 as cv

## Image ##
from skimage import exposure # re-balance images exposure and luminosity

## Environment import ##
from dotenv import load_dotenv

###############
## Constants ##
###############

## Environment setup ##
load_dotenv()
USER = os.getenv('USER')
PASSWORD = os.getenv('PASSWORD')
URL = os.getenv('URL')

#############
## Classes ##
#############

class Query():
	"""
	# Query class Content #
	Object used to extract data online, from Copernicus Open Access Hub with sentinelsat API and module.
	
	Attributes:
		- geojson: String, name of the json file containing the footprint data
		- date_range: datetime.date() tuple, time boundaries of the query
		- cloudcoverpercentage: float tuple, cloud cover percentage boundaries of the query
		- q_dataframe: DataFrame, query answer retrieved from Sentinel API as a DataFrame
	
	Properties:
		- length: int, size of q_dataframe (an empty DataFrame means that no products matched the query)
		- date_from: datetime.date(), low time boundary for the query
		- date_to: datetime.date(), high time boundary for the query

	Constants:
		- SOONER/LATER: int, these 2 constants represent the 2 possible modes for the query process.
		- DATESTEPS: list(String), this constant is a list with every possible steps for the query process.
	"""
	
	SOONER = 1
	LATER = 2
	DATESTEPS = ["day","week","month"]

	def __init__(self, geojson, date_range, cloudcoverpercentage):
		self.api = self.connect()
		self.geojson = os.path.normpath(os.getcwd() + os.sep + "GeoJSON" + os.sep + geojson)
		self.date_range = date_range
		self.cloudcoverpercentage = cloudcoverpercentage
		self.q_dataframe = None

	@property
	def length(self):
		return self.q_dataframe.shape[0]

	@property
	def date_from(self):
		return self.date_range[0]

	@property
	def date_to(self):
		return self.date_range[1]

	@classmethod
	def connect(cls):
		"""
		Connect to the API using credentials stored into .env file.
		"""
		try:
			print("="*40, flush=True)
			print("Connected to Sentinel API.", flush=True)
			print("="*40, flush=True)
			return SentinelAPI(USER, PASSWORD, URL)
		except Exception as e:
			raise e


	def query_from_json(self):
		"""
		Send a query using a geoJSON.
		"""
		footprint = geojson_to_wkt(read_geojson(self.geojson))
		products = self.api.query(
								  footprint,
						 		  date=self.date_range,
								  platformname="Sentinel-2",
								  processinglevel="Level-2A",
								  cloudcoverpercentage=(0, self.cloudcoverpercentage),
								  area_relation="Contains",
						 		  order_by="cloudcoverpercentage"
						 		  )
		self.q_dataframe = self.api.to_dataframe(products)

	def extend_date_range(self,mode,step):
		"""
		2 modes available:
			- mode = Query.SOONER, i.e. 1, date_from is set 
			- mode = Query.LATER, i.e. 2, date_from is set 
		step is a string: step="day"/"week"/"month"
		"""
		if not self.DATESTEPS.__contains__(step):
			print("Step format error. Setting step to 'week'...", flush=True)
			step = "week" #Set week as default step

		steps_to_days = {"day":1, "week":7, "month":30}
		timedelta = td(days=steps_to_days[step])

		terminated = False
		while not terminated:
			if mode == self.SOONER:
				new_date_from = self.date_range[0]-timedelta
				new_date_to = self.date_range[0]
				self.date_range = (new_date_from,new_date_to)
				print(self.date_range)
				terminated = True
			elif mode == self.LATER:
				new_date_from = self.date_range[1]
				new_date_to = self.date_range[1]+timedelta
				self.date_range = (new_date_from, new_date_to)
				print(self.date_range)
				terminated = True
			else:
				mode = input("Choose a correct mode (1=SOONER, 2=LATER) : ")
		print("Date range changed.", flush=True)
		
	def find_data(self, mode, step):
		self.query_from_json()
		while (self.length < 1):
			print("No product found. Retrying with a new date range.", flush=True)
			self.extend_date_range(mode, step)
			self.query_from_json()
		return self.q_dataframe.head(1).index[0], self.q_dataframe.head(1)["cloudcoverpercentage"] #return the name of the first file and cloud cover percentage

	def download_item(self, name):
		"""
		Download a single item from the query.
		"""
		directory = os.path.normpath(os.getcwd() + os.sep + "Downloads")
		meta = self.api.download(name, directory_path=directory)
		return meta

	def unzip_data(self, meta):
		zip_dir = meta["path"]
		data_dir = os.path.normpath(os.getcwd() + os.sep + "Data")
		with zipfile.ZipFile(zip_dir, "r") as zip_ref:
			zip_ref.extractall(data_dir)
		print("## Data unzipped ##")

	def process(self, mode, step):
		"""
		"""
		_bool = True
		count = 0
		while _bool:
			name, cc = self.find_data(mode=mode, step=step)
			ans = input("File found."+"\n"+"Name and cloud cover percentage: {}".format(cc)+"\n"+"Do you want to download it? [y/n]\n")
			if ans == "y":
				meta = self.download_item(name)
				_bool = False
				print("## Data succesfully retrieved ##", flush=True)
				self.unzip_data(meta)
				img_writer = ImageWriter(meta, self.geojson)
				img_writer.write()
			else:
				self.extend_date_range(mode=mode, step=step)
			count += 1
			if count == 20:
				ans = input("Do you want to keep going? [y/n]")
				if ans == "n":
					_bool = False
				else:
					count = 0

class ImageWriter():
	"""
	"""

	WORKING_PATH = os.path.dirname(__file__)
	DATA_PATH = os.path.join(WORKING_PATH, "Data")
	ARCHIVES_PATH = os.path.join(WORKING_PATH, "Archives")
	IMAGES_PATH = os.path.join(WORKING_PATH, "Images")

	def __init__(self, meta, geojson):
		self.meta = meta
		self.geojson = geojson
		print(f"Current working directory: {self.WORKING_PATH}", flush=True)

	def get_data_path(self):
		"""
		Return IMG_DATA folder path. This folder contains 3 folders, one for each resolution (10m, 20m and 60m).
		"""
		path = os.path.normpath(self.DATA_PATH + os.sep + self.meta["title"] + ".SAFE" + os.sep + "GRANULE")
		path = os.path.normpath(path + os.sep + os.listdir(path)[0] + os.sep + "IMG_DATA")
		return path

	def create_dir_path(self):
		"""
		Create the directory folders where images will be written. Return the new directory.
		"""
		folder_name = self.meta["title"]
		new_path = os.path.join(self.ARCHIVES_PATH, folder_name)
		os.mkdir(new_path) # Create new folder where images will be stored

		self.meta["date"] = self.meta["date"].strftime("%Y %m %d %H:%M")
		self.meta["Creation Date"] = self.meta["Creation Date"].strftime("%Y %m %d %H:%M")
		self.meta["Ingestion Date"] = self.meta["Ingestion Date"].strftime("%Y %m %d %H:%M")
		
		## Create meta file ##
		with open(new_path+"\\meta.json","w") as f_meta:
			json.dump(self.meta,f_meta,indent=4)
			f_meta.close()

		## Create resolution folders ##
		os.mkdir(new_path + os.sep + "R10m")
		os.mkdir(new_path + os.sep + "R20m")
		os.mkdir(new_path + os.sep + "R60m")

		return new_path

	def crop_image(self, source_file, dest_file):
		"""
		Crop an image to keep only the area of interest and write it into destination file. Source file is a temporary file containing all bands data.
		Source file is erased at the end of the process.
		"""
		with rio.open(source_file) as src:
			# create shape from geojson
			with fiona.open(self.geojson) as geojson:
				features = [feature["geometry"] for feature in geojson]
				new_geom = []
				for geom in features:
					#apply the good crs to the geometry
					new_geom.append(tf.transform_geom(geojson.crs,src.crs.to_dict(),geom))

			out_image, out_transform = msk.mask(src, new_geom, crop=True)
			out_meta = src.meta.copy()
			out_meta.update({"driver": "Gtiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
			src.close()
		os.remove(source_file) # Erase temporary source file

		with rio.open(dest_file, 'w', **out_meta) as dest:
			dest.write(out_image)
			dest.close()
		print("## New cropped image created ##", flush=True)

	def write(self):
		data_path = self.get_data_path()
		dir_path = self.create_dir_path()
		list_res = ["R10m","R20m","R60m"]
		for res in list_res:
			self.write_res(data_path, res, dir_path)
		print("## Image writen ##", flush=True)

	def write_res(self, data_path, res, dir_path):
		"""
		'res' argument is a String with the value "R10m", "R20m" or "R60m".
		"""
		new_data_path = os.path.join(data_path, res)
		new_dir_path = os.path.join(dir_path, res)

		image_name = self.meta["title"] + ".tif"
		info = {"filename":image_name,"bands":{}}
		list_bands = os.listdir(new_data_path)

		band_number = 1

		b = rio.open(os.path.join(new_data_path, list_bands[1]), 'r')
		with rio.open(os.path.join(self.WORKING_PATH, "temp" + os.sep + "temp.tiff"), 'w', driver="Gtiff", width=b.width, height=b.height, count=len(list_bands), crs=b.crs, transform=b.transform, dtype=b.dtypes[0]) as temp:
			b.close()
			b = None
			for f in list_bands:
				ftype = f.split("_")[2]
				bool = ((ftype != "AOT") and (ftype != "SCL") and (ftype != "TCI") and (ftype != "WVP"))
				if (bool):
					with rio.open(os.path.join(new_data_path,f),'r') as band:
						print(f"## Writing band {ftype} for resolution {res} in temporary file. ##", flush=True)
						temp.write(band.read(1), band_number)
						info["bands"][ftype] = band_number
						band_number += 1
						band.close()
			temp.close()

		self.crop_image(os.path.join(self.WORKING_PATH, "temp" + os.sep + "temp.tiff"), os.path.join(new_dir_path, image_name))
		
		with open(os.path.join(new_dir_path, "bands_info.json"), 'w') as foo:
			json.dump(info, foo, indent=4)
			foo.close()

class Image():
	"""
	# Image class content #
	Object used to get images from the data previously downloaded.

	Attributes:
		- name: String, name of the folder where the wanted image is stored.
		- folder_path: String, path of the folder where the image is stored.
		- meta: dict(), metadata of the image.
		- image: np.array(), image stored as an array. None if no image has been loaded.

	Constants:
		- BANDS: dict(), give the possible bands and their name for each resolution.
		- WORKING_PATH: String, current working directory.
		- ARCHIVES_PATH: String, path where all archives are stored, i.e. downloaded and pre-processes images.

	"""
	
	BANDS = {
			 "10m": {"B02":"Blue", "B03":"Green", "B04":"Red", "B08":"NIR"},
			 "20m": {"B02":"Blue", "B03":"Green", "B04":"Red", "B05":"Re1", "B06":"Re2", "B07":"Re3", "B11":"SWIR1", "B12":"SWIR2", "B8A":"NIRn"},
			 "60m": {"B01":"Coastal aerosol", "B02":"Blue", "B03":"Green", "B04":"Red", "B05":"Re1", "B06":"Re2", "B07":"Re3", "B09":"Water vapor", "B11":"SWIR1", "B12":"SWIR2", "B8A":"NIRn"}
			 }
	WORKING_PATH = os.path.dirname(__file__)
	ARCHIVES_PATH = os.path.join(WORKING_PATH, "Archives")
	IMAGES_PATH = os.path.join(WORKING_PATH, "Images")

	def __init__(self, name):
		# Files attributes
		self.name = name # Name of the .tif file
		self.folder_path = os.path.join(self.ARCHIVES_PATH, name) # Folder path where files are stored for each resolution
		with open(os.path.join(self.folder_path, "meta.json"), 'r') as foo:
			self.meta = json.load(foo) # Metadata
			foo.close()

		# Image attributes
		self.image = None # Array with shape (x, y, 3) if it's a 3 band combination or (x, y) if it's a computed index
		self.is_normalized = False

	def get_date(self):
		"""
		Return the date when image was shot as a String.
		"""
		return self.meta["date"]

	def load_info(self, res):
		res_path = os.path.join(self.folder_path, res)
		with open(os.path.join(res_path, "bands_info.json"), 'r') as foo:
			b_info = json.load(foo)
			foo.close()
		return res_path, b_info

	def load_image_bands(self, path, filename, bands):
		"""
		Load a 3-band image.

		"bands" is a list representing the band combination of the image. 
		"""
		if len(bands) != 3:
			print("## Error. Wrong band combination.", flush=True)
		else:
			with rio.open(os.path.join(path, filename), 'r') as file:
				self.image =  plot.reshape_as_image(
													[file.read(bands[0]),
													 file.read(bands[1]),
													 file.read(bands[2])]
													)
				file.close()

	def load_image_indexes(self, path, filename, indexes):
		"""
		Load the image file and return the array given by: (indexes[0]-indexes[1])/(indexes[0]+indexes[1]).

		"indexes" is a list containing names of the bands used to calculate the index.
		"""
		if len(indexes) != 2:
			print("## Error. Wrong band combination.", flush=True)
		else:
			with rio.open(os.path.join(path, filename), 'r') as file:
				arr1 = file.read(indexes[0])
				arr2 = file.read(indexes[1])
				self.image = (arr1.astype("float32")-arr2.astype("float32"))/(arr1+arr2)
				file.close()

	def load_single_band(self, res, band):
		"""
		Open the image file and return an array containing data from one single band.

		The purpose of this function is to extract data from one specific band, without any normalization applied.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if not b_info["bands"].keys().__contains__(band):
			print(f"Band {band} not available for this resolution.", flush=True)
		else:
			with rio.open(os.path.join(res_path, filename), 'r') as file:
				self.image = file.read(b_info["bands"][band])
				file.close()
		return self.image

	def get_RGB(self, res, pLow, pHigh):
		"""
		True Color Image.
		Get the Red-Green-Blue band combination image and return it.
		
		"pLow" and "pHigh" enable to change the exposure of the image.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		bands = [b_info["bands"]["B04"], b_info["bands"]["B03"], b_info["bands"]["B02"]]
		self.load_image_bands(res_path, filename, bands)
		self.linearStretch(pLow, pHigh)
		self.normalize_image()
		return self.image

	def get_FIR(self, res, pLow, pHigh):
		"""
		False-Infrared Color Image.
		Get the NIR-Red-Green band combination image and return it.
		Healthy vegetation is seen bright red, clear water is seen dark blue... 

		"pLow" and "pHigh" enable to change the exposure of the image.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			bands = [b_info["bands"]["B08"], b_info["bands"]["B04"], b_info["bands"]["B03"]]
		else:
			bands = [b_info["bands"]["B8A"], b_info["bands"]["B04"], b_info["bands"]["B03"]]
		self.load_image_bands(res_path, filename, bands)
		self.linearStretch(pLow, pHigh)
		self.normalize_image()
		return self.image
	
	def get_Agriculture(self, res, pLow, pHigh):
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			print("## Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions. ##", flush=True)
		else:
			bands = [b_info["bands"]["B11"], b_info["bands"]["B8A"], b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			self.linearStretch(pLow,pHigh)
			self.normalize_image()
			return self.image

	def get_FCUrban(self, res, pLow, pHigh):
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			print("## Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions. ##", flush=True)
		else:
			bands = [b_info["bands"]["B12"], b_info["bands"]["B11"], b_info["bands"]["B04"]]
			self.load_image_bands(res_path, filename, bands)
			self.linearStretch(pLow, pHigh)
			self.normalize_image()
			return self.image

	def get_Geology(self, res, pLow, pHigh):
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			print("## Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions. ##", flush=True)
		else:
			bands = [b_info["bands"]["B12"], b_info["bands"]["B11"], b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			self.linearStretch(pLow, pHigh)
			self.normalize_image()
			return self.image

	def get_HVeg(self, res, pLow, pHigh):
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			print("## Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions. ##", flush=True)
		else:
			bands = [b_info["bands"]["B8A"],b_info["bands"]["B11"],b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			self.linearStretch(pLow, pHigh)
			self.normalize_image()
			return self.image

	def get_Coastal(self, res, pLow, pHigh):
		"""
		Coastal RGB-like Color Image.
		Get an image close to the RGB band combination, but improved for coastal areas.

		"pLow" and "pHigh" enable to change the exposure of the image.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if (res == "R10m" or res == "R20m"):
			print("## Image not available for this resolution. Please choose 'R60m' resolution. ##", flush=True)
		else:
			bands = [b_info["bands"]["B04"], b_info["bands"]["B03"], b_info["bands"]["B01"]]
			self.load_image_bands(res_path, filename, bands)
			self.linearStretch(pLow, pHigh)
			self.normalize_image()
			return self.image

	def get_NDVI(self,res):
		"""
		Normalized Difference Vegetation Index.
		Compute the NDVI as an array and return it.
		Useful to monitor droughts, urban expansion, land use in general.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			indexes = [b_info["bands"]["B08"],b_info["bands"]["B04"]]
		else:
			indexes = [b_info["bands"]["B8A"],b_info["bands"]["B04"]]
		self.load_image_indexes(res_path, filename, indexes)
		return self.image

	def get_NDWIveg(self,res):
		"""
		Normalized Difference Water Index - Vegetation.
		Compute the NDWIveg as an array and return it.
		Display the difference of water content in vegetation.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			print("## Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions. ##", flush=True)
		else:
			indexes = [b_info["bands"]["B8A"], b_info["bands"]["B12"]]
			self.load_image_indexes(res_path, filename, indexes)
			return self.image

	def get_NDWIwb(self, res):
		"""
		Normalized Difference Water Index - Water Bodies.
		Compute the NDWIwb as an array and return it.
		Useful to identify water.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			indexes = [b_info["bands"]["B03"], b_info["bands"]["B08"]]
		else:
			indexes = [b_info["bands"]["B03"], b_info["bands"]["B8A"]]
		self.load_image_indexes(res_path, filename, indexes)
		return self.image

	def is_loaded(self):
		"""
		Check if current image has been loaded. Return False if not.
		"""
		return (self.image != None).any()

	def linearStretch(self, pLow, pHigh):
		"""
		Apply linear stretching to the current image. Thus, it enlightens the image.
		"""

		if not self.is_loaded:
			print("## Error: no image previously extracted. Could not proceed to stretching. ##")
		else:
			print("## Proceeding to stretching. ##")
			tmp_image = []
			for i in range(self.image.shape[np.argmin(self.image.shape)]):
				band = self.image[:,:,i]
				iMin, iMax = np.percentile(band[~np.isnan(band)], (pLow, 100-pHigh))
				band_rescaled = exposure.rescale_intensity(band, in_range=(iMin, iMax))
				tmp_image.append(band_rescaled)
			img_rescale = np.dstack(tmp_image)

			self.image = img_rescale

	def normalize_image(self):
		"""
		Normalize the current image, set values between 0 and 255 for every pixel.
		"""
		if not self.is_loaded():
			print("## Error: no image previously extracted. Could not proceed to normalization. ##", flush=True)
		else:
			if self.is_normalized:
				print("## Image already normalized. ##", flush=True)
			else:
				print("## Proceeding to normalization. ##", flush=True)
				tab = self.image
				tab = 255*((tab-tab.min())/(tab.max()-tab.min()))
				tab[tab<0] = 0
				tab[tab>255] = 255
				tab[~np.isfinite(tab)] = 0
				self.image = tab.astype("uint16")

	def save(self, filename : str):
		"""
		Save the current image into a .png file.
		"""
		if not self.is_loaded():
			print("No image loaded.", flush=True)
		else:
			print("Saving image...", flush=True)
			if check_image_filename(filename):
				cv.imwrite(os.path.join(self.IMAGES_PATH, filename), self.image)
			else:
				print("Image filename not valid!", flush=True)

###############
## Functions ##
###############

def check_image_filename(filename):
	"""
	Return True if filename needs to be changed.
	"""
	return not ((filename == "") or (filename[len(filename)-4:len(filename)] != ".png") or (os.listdir(Image.IMAGES_PATH).__contains__(filename)))

def main():
	q = Query("Amboise.json", (date(2021, 4, 30), date(2021, 5, 1)), 10)
	q.process(1, "day")

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.")