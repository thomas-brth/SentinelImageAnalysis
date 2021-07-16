# Sentinel-2 satellite data extraction and processing script

#############
## Imports ##
#############

## Sentinelsat imports ##
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt #create and retrieve a query

## General imports ##
import os
import sys
sys.path.append("utils")
sys.path.append("utils\\processing")
import json # for metadata
import zipfile # unzip collected data
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

## Rasterio imports ##
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
from skimage import exposure, morphology

## Environment import ##
from dotenv import load_dotenv

## Logger ##
import logging

## Custom imports ##
from utils.processing import crop_2D, crop_3D

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
	"""
	
	SOONER = 1
	LATER = 2
	DATESTEPS = ["day","week","month"]

	def __init__(self, geojson : str, date_range : tuple, cloudcoverpercentage : int):
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

	def extend_date_range(self, mode, step):
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
		"""
		Search for available and corresponding data. Return the id of the found tile with the best cloudcover percentage.
		"""
		self.query_from_json()
		while (self.length < 1):
			print("No product found. Retrying with a new date range.", flush=True)
			self.extend_date_range(mode, step)
			self.query_from_json()
		return self.q_dataframe.head(1).index[0], self.q_dataframe.head(1)["cloudcoverpercentage"]

	def download_item(self, name : str):
		"""
		Download a single item from the query.
		"""
		directory = os.path.normpath(os.getcwd() + os.sep + "Downloads")
		meta = self.api.download(name, directory_path=directory)
		return meta

	def unzip_data(self, meta : dict):
		"""
		Unzip downloaded data.
		"""
		zip_dir = meta["path"]
		data_dir = os.path.normpath(os.getcwd() + os.sep + "Data")
		with zipfile.ZipFile(zip_dir, "r") as zip_ref:
			zip_ref.extractall(data_dir)
		print("## Data unzipped ##")

	def process(self, mode, step):
		"""
		Proceed to retreiving desired data. 
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
	# ImageWriter class Content #
	Object used to write downloaded data into desired format. Data are then stored in separate folders, ordered by resolution.
	
	Attributes:
		- meta: Dictionary, metadata of downloaded data
		- geojson: String, name of the json file containing the footprint data
	"""

	WORKING_PATH = os.path.dirname(__file__)
	DATA_PATH = os.path.join(WORKING_PATH, "Data")
	ARCHIVES_PATH = os.path.join(WORKING_PATH, "Archives")
	IMAGES_PATH = os.path.join(WORKING_PATH, "Images")

	def __init__(self, meta : dict, geojson : str):
		# Logger
		self.logger = logging.getLogger("IMGWR LOG")
		self.logger.setLevel(logging.DEBUG)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s : %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S"))
		self.logger.addHandler(ch)

		self.meta = meta
		self.geojson = geojson
		self.logger.debug(f"Current working directory: {self.WORKING_PATH}")

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
					# apply the good crs to the geometry
					new_geom.append(tf.transform_geom(geojson.crs, src.crs.to_dict(), geom))

			out_image, out_transform = msk.mask(src, new_geom, crop=True)
			out_meta = src.meta.copy()
			out_meta.update({"driver": "Gtiff", "height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform})
			src.close()
		os.remove(source_file) # Erase temporary source file

		with rio.open(dest_file, 'w', **out_meta) as dest:
			dest.write(out_image)
			dest.close()
		self.logger.debug("New cropped image created.")

	def write(self):
		"""
		Write data into a .tiff file for each resolution.
		"""
		data_path = self.get_data_path()
		dir_path = self.create_dir_path()
		list_res = ["R10m","R20m","R60m"]
		for res in list_res:
			self.write_res(data_path, res, dir_path)
		self.logger.debug("Image writen.")

	def write_res(self, data_path : str, res : str, dir_path : str):
		"""
		Write data for a given resolution.
		"""
		new_data_path = os.path.join(data_path, res)
		new_dir_path = os.path.join(dir_path, res)

		image_name = self.meta["title"] + ".tif"
		info = {"filename":image_name,"bands":{}}
		list_bands = os.listdir(new_data_path)

		band_number = 1

		b = rio.open(os.path.join(new_data_path, list_bands[1]), 'r')
		_type = 'uint16'
		with rio.open(os.path.join(self.WORKING_PATH, "temp" + os.sep + "temp.tiff"), 'w', driver="Gtiff", width=b.width, height=b.height, count=len(list_bands), crs=b.crs, transform=b.transform, dtype=_type) as temp:
			b.close()
			b = None
			for f in list_bands:
				ftype = f.split("_")[2]
				_bool = ((ftype != "AOT") and (ftype != "TCI") and (ftype != "WVP"))
				if (_bool):
					with rio.open(os.path.join(new_data_path, f),'r') as band:
						self.logger.debug(f"Writing {ftype} for resolution {res} in temporary file.")
						temp.write(band.read(1).astype(_type), band_number)
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
	"""
	
	WORKING_PATH = os.path.dirname(__file__)
	ARCHIVES_PATH = os.path.join(WORKING_PATH, "Archives")
	IMAGES_PATH = os.path.join(WORKING_PATH, "Images")

	def __init__(self, name : str):
		# Logger
		self.logger = logging.getLogger("IMG LOG")
		self.logger.setLevel(logging.DEBUG)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s : %(levelname)s - %(message)s", datefmt="%m/%d/%Y %I:%M:%S"))
		self.logger.addHandler(ch)

		# Files attributes
		self.name = name # Name of the folder
		self.folder_path = os.path.join(self.ARCHIVES_PATH, self.name) # Folder path where files are stored for each resolution
		with open(os.path.join(self.folder_path, "meta.json"), 'r') as foo:
			self.meta = json.load(foo) # Metadata
			self.logger.debug(f"Image {self.name} loaded.")
			self.logger.debug(f"Image acquired on {self.meta['date']}.")
			foo.close()

		# Image attributes
		self.image = None # Array with shape (x, y, 3) if it's a 3 band combination or (x, y) if it's a computed index
		self.pproc = {
			"crop" : False,
			"stretch" : True,
			"pLow" : 0.05,
			"pHigh" : 0.85,
			"normalize": True
			}

	@property
	def date(self):
		"""
		Return the date when data was acquired.
		"""
		return self.meta["date"]

	#####################
	## Generic methods ##
	#####################

	def set_pproc_settings(self, **kwargs):
		"""
		Update pre-processing settings.
		"""
		self.pproc.update(kwargs)

	def load_info(self, res : str):
		"""
		Load band metadata for a given resolution.
		"""
		res_path = os.path.join(self.folder_path, res)
		with open(os.path.join(res_path, "bands_info.json"), 'r') as foo:
			b_info = json.load(foo)
			foo.close()
		return res_path, b_info

	def is_loaded(self):
		"""
		Check if current image has been loaded. Return False if not.
		"""
		return (self.image != None).any()

	def linear_stretch(self):
		"""
		Apply linear stretching to the current image.
		"""
		self.logger.debug("Proceeding to linear stretching.")
		tmp_image = []
		for i in range(self.image.shape[np.argmin(self.image.shape)]):
			band = self.image[:,:,i]
			iMin, iMax = np.percentile(band[~np.isnan(band)], (self.pproc["pLow"], 100-self.pproc["pHigh"]))
			band_rescaled = exposure.rescale_intensity(band, in_range=(iMin, iMax))
			tmp_image.append(band_rescaled)
		img_rescale = np.dstack(tmp_image)

		self.image = img_rescale

	def _normalize_image(self):
		"""
		Normalize the current image, set values between 0 and 255 for every pixel.
		"""
		self.logger.debug("Proceeding to normalization.")
		arr = self.image
		arr = 255 * ((arr - arr.min()) / (arr.max() - arr.min()))
		arr[arr < 0] = 0
		arr[arr > 255] = 255
		arr[~np.isfinite(arr)] = 0
		self.image = arr.astype("uint16")

	def save(self, filename : str, colormap : int = cv.COLORMAP_MAGMA):
		"""
		Save the current image into a .png file.
		"""
		if not self.is_loaded():
			self.logger.warning("No image loaded.")
		else:
			self.logger.debug("Saving image...")
			if check_image_filename(filename):
				if self.image.ndim == 3:
					cv.imwrite(os.path.join(self.IMAGES_PATH, filename), self.image.astype("uint8"))
				else:
					self.image *= 255
					cv.imwrite(os.path.join(self.IMAGES_PATH, filename), cv.applyColorMap(self.image.astype("uint8"), colormap))
			else:
				self.logger.error("Image filename not valid!")

	def load_image_bands(self, path : str, filename : str, bands : list):
		"""
		Load a 3-band image.

		"bands" is a list representing the band combination of the image. 
		"""
		if len(bands) != 3:
			self.logger.error("Not enough bands to load into an image.")
		else:
			with rio.open(os.path.join(path, filename), 'r') as file:
				self.image =  plot.reshape_as_image(
													[
													file.read(bands[0]),
													file.read(bands[1]),
													file.read(bands[2])
													]
													)
				file.close()
			# Post-processing applied to the image
			if self.pproc["crop"]:
				self.image = crop_3D(self.image)
			if self.pproc["stretch"]:
				self.linear_stretch()
			if self.pproc["normalize"]:
				self._normalize_image()

	def load_single_band(self, res : str, band : str, factor : float = 1):
		"""
		Open the image file and return an array containing data from one single band.

		A factor can be applied.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if not b_info["bands"].keys().__contains__(band):
			self.logger.error(f"Band {band} not available for this resolution.")
		else:
			with rio.open(os.path.join(res_path, filename), 'r') as file:
				self.image = file.read(b_info["bands"][band]) * factor
				file.close()

			# Post-processing applied to the band
			if self.pproc["crop"]:
				self.image = crop_2D(self.image)

		return self.image

	def compute_NDI(self, res : str, band_1 : str, band_2 : str):
		"""
		Compute a Normalize Difference Index, i.e. the difference between 2 bands, divided by their sum.
		Values varies between -1 and 1
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if not b_info["bands"].keys().__contains__(band_1) or not b_info["bands"].keys().__contains__(band_2):
			self.logger.error(f"Bands not available for this resolution.")
		else:
			with rio.open(os.path.join(res_path, filename), 'r') as file:
				if self.pproc['crop']:
					arr_1 = crop_2D(file.read(b_info["bands"][band_1]))
					arr_2 = crop_2D(file.read(b_info["bands"][band_2]))
				else:
					arr_1 = file.read(b_info["bands"][band_1])
					arr_2 = file.read(b_info["bands"][band_2])
				self.image = (arr_1.astype("float32") - arr_2.astype("float32")) / (arr_1 + arr_2)
				file.close()

	def compute_ratio(self, res : str, band_1 : str, band_2 : str):
		"""
		Compute a simple ratio between 2 bands.
		Non-finite values are replaced by zeros.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if not b_info["bands"].keys().__contains__(band_1) or not b_info["bands"].keys().__contains__(band_2):
			self.logger.error(f"Bands not available for this resolution.")
		else:
			with rio.open(os.path.join(res_path, filename), 'r') as file:
				if self.pproc['crop']:
					arr_1 = crop_2D(file.read(b_info["bands"][band_1]))
					arr_2 = crop_2D(file.read(b_info["bands"][band_2]))
				else:
					arr_1 = file.read(b_info["bands"][band_1])
					arr_2 = file.read(b_info["bands"][band_2])
				self.image = arr_1.astype("float32") / arr_2.astype("float32")
				self.image[~np.isfinite(self.image)] = 0
				file.close()

	#####################
	## Special methods ##
	#####################

	def get_SCL(self, res : str, crop : bool = False):
		"""
		Get the scene classification image, as processed by Sentinel L2A SC Algorithm.
		"""
		if res == "R10m":
			# Get 20m resolution and simply double the number of grid points. 
			scl = self.load_single_band(res="R20m", band="SCL")
			n, m = scl.shape
			new_scl = np.zeros((n * 2, m * 2))
			for i in range(n):
				for j in range(m):
					new_scl[i * 2, j * 2] = scl[i, j]
					new_scl[i * 2 + 1, j * 2] = scl[i, j]
					new_scl[i * 2, j * 2 + 1] = scl[i, j]
					new_scl[i * 2 + 1, j * 2 + 1] = scl[i, j]
			self.image = new_scl
		else:
			self.load_single_band(res=res, band="SCL")
		return self.image

	def get_RGB(self, res : str):
		"""
		True Color Image.
		Get the Red-Green-Blue band combination image and return it.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		bands = [b_info["bands"]["B04"], b_info["bands"]["B03"], b_info["bands"]["B02"]]
		self.load_image_bands(res_path, filename, bands)
		return self.image

	def get_FIR(self, res : str):
		"""
		False-Infrared Color Image.
		Get the NIR-Red-Green band combination image and return it.
		Healthy vegetation is seen bright red, clear water is seen dark blue... 
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			bands = [b_info["bands"]["B08"], b_info["bands"]["B04"], b_info["bands"]["B03"]]
		else:
			bands = [b_info["bands"]["B8A"], b_info["bands"]["B04"], b_info["bands"]["B03"]]
		self.load_image_bands(res_path, filename, bands)
		return self.image
	
	def get_Agriculture(self, res : str):
		"""
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			self.logger.warning("Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions.")
		else:
			bands = [b_info["bands"]["B11"], b_info["bands"]["B8A"], b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			return self.image

	def get_FCUrban(self, res : str):
		"""
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			self.logger.warning("Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions.")
		else:
			bands = [b_info["bands"]["B12"], b_info["bands"]["B11"], b_info["bands"]["B04"]]
			self.load_image_bands(res_path, filename, bands)
			return self.image

	def get_Geology(self, res : str):
		"""
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			self.logger.warning("Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions.")
		else:
			bands = [b_info["bands"]["B12"], b_info["bands"]["B11"], b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			return self.image

	def get_HVeg(self, res : str):
		"""
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if res == "R10m":
			self.logger.warning("Image not available for this resolution. Please choose 'R20m' or 'R60m' resolutions.")
		else:
			bands = [b_info["bands"]["B8A"], b_info["bands"]["B11"], b_info["bands"]["B02"]]
			self.load_image_bands(res_path, filename, bands)
			return self.image

	def get_Coastal(self, res : str):
		"""
		Coastal RGB-like Color Image.
		Get an image close to the RGB band combination, but improved for coastal areas.
		"""
		res_path, b_info = self.load_info(res)
		filename = b_info["filename"]
		if (res == "R10m" or res == "R20m"):
			self.logger.warning("Image not available for this resolution. Please choose 'R60m' resolution.")
		else:
			bands = [b_info["bands"]["B04"], b_info["bands"]["B03"], b_info["bands"]["B01"]]
			self.load_image_bands(res_path, filename, bands)
			return self.image

	def get_NDVI(self, res : str):
		"""
		Normalized Difference Vegetation Index.
		Compute the NDVI as an array and return it.
		Useful to monitor droughts, urban expansion, land use in general.
		"""
		if res == "R10m":
			self.compute_NDI(res=res, band_1="B08", band_2="B04")
		else:
			self.compute_NDI(res=res, band_1="B8A", band_2="B04")
		return self.image

	def get_NDWIveg(self, res : str):
		"""
		Normalized Difference Water Index - Vegetation.
		Compute the NDWIveg as an array and return it.
		Display the difference of water content in vegetation.
		"""
		if res == "R10m":
			self.logger.warning("Bands not available for this resolution.")
		self.compute_NDI(res=res, band_1="B8A", band_2="B12")
		return self.image

	def get_NDWIwb(self, res : str):
		"""
		Normalized Difference Water Index - Water Bodies.
		Compute the NDWIwb as an array and return it.
		Useful to identify water.
		"""
		if res == "R10m":
			self.compute_NDI(res=res, band_1="B03", band_2="B08")
		else:
			self.compute_NDI(res=res, band_1="B03", band_2="B8A")
		return self.image

###############
## Functions ##
###############

def check_image_filename(filename):
	"""
	Return True if filename needs to be changed.
	"""
	return not ((filename == "") or ((filename[len(filename)-4:len(filename)] != ".png") and (filename[len(filename)-4:len(filename)] != ".jpg")) or (os.listdir(Image.IMAGES_PATH).__contains__(filename)))

def main():
	q = Query("Lytton.json", (date(2021, 7, 4), date(2021, 7, 5)), 50)
	q.process(1, "day")

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.")