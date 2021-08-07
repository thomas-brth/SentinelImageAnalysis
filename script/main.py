# Command line script

#############
## Imports ##
#############

## General imports ##
import argparse
import os
import sys
sys.path.append("utils")
sys.path.append("utils\\processing")
from datetime import date
import logging

## Custom imports ##
from image import Query

###############
## Constants ##
###############

#############
## Classes ##
#############

class SentinelParser(argparse.ArgumentParser):
	"""
	"""
	def __init__(self, **kwargs):
		super(SentinelParser, self).__init__(**kwargs)
		# Logger
		self.logger = logging.getLogger("PARSER LOG")
		self.logger.setLevel(logging.DEBUG)
		ch = logging.StreamHandler()
		ch.setLevel(logging.DEBUG)
		ch.setFormatter(logging.Formatter(fmt="%(asctime)s - %(name)s : %(levelname)s - %(message)s", datefmt="%d/%m/%Y %I:%M:%S"))
		self.logger.addHandler(ch)

		# Add all the arguments
		self.add_argument('-f', '--filename', help="Name of the geoJSON file, without the extension.", nargs=1, type=str, required=True)
		self.add_argument('-df', '--date-from', help="Starting date for the query. Format : YYYY MM DD", nargs=3, type=int, required=True)
		self.add_argument('-dt', '--date-to', help="Ending date for the query. Format : YYYY MM DD", nargs=3, type=int, required=True)
		self.add_argument('-cc', '--cloudcover-percentage', help="Maximum acceptable cloud cover percentage.", nargs=1, type=int, required=True)
		self.add_argument('-n', '--name', help="Give a name to your downloaded image. It will be the image's folder name.", nargs=1, type=str, required=True)
		self.add_argument('-s', '--search-type', help="Type 's' for a single query and 'i' for an iterative one.", nargs=1, type=str, required=False, default="s")
		self.add_argument('-ds', '--day-step', help="Shift the date of the given number of days for an iterative query.", nargs=1, type=int, required=False, default=5)
		self.add_argument('-i', '--iter-max', help="Specify the maximum number of iterations for an iterative query.", nargs=1, type=int, required=False, default=50)
		self.add_argument('-b', '--backward', help="Type '0' for a backward search and '1' for a forward one.", nargs=1, type=int, required=False, default=0)

		# Parse arguments
		self.args = self.parse_args()

	@property
	def filename(self):
		"""
		"""
		return self.args.filename[0] + ".json"

	@property
	def date_from(self):
		"""
		"""
		return date(self.args.date_from[0], self.args.date_from[1], self.args.date_from[2])

	@property
	def date_to(self):
		"""
		"""
		return date(self.args.date_to[0], self.args.date_to[1], self.args.date_to[2])

	@property
	def cloudcover_percentage(self):
		"""
		"""
		return self.args.cloudcover_percentage[0]

	@property
	def name(self):
		"""
		"""
		return self.args.name[0]

	@property
	def search_type(self):
		"""
		"""
		return self.args.search_type[0]

	@property
	def day_step(self):
		"""
		"""
		return self.args.day_step[0]

	@property
	def iter_max(self):
		"""
		"""
		return self.args.iter_max[0]

	@property
	def backward(self):
		"""
		"""
		return self.args.backward[0]

	def _check_arguments(self):
		"""
		"""
		# Check if geoJSON filename is valid
		if not self.filename in os.listdir(os.path.normpath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "GeoJSON"))):
			self.logger.error("GeoJSON file not found.")
			raise Exception()

		# Check if date range is valid
		if not (self.date_from < self.date_to or self.date_to < date.today()):
			self.logger.error("Date range not valid.")
			raise Exception()

		# Check if clod cover percentage is valid
		if self.cloudcover_percentage <= 0 or self.cloudcover_percentage > 100:
			self.logger.error("Cloud cover percentage not valid.")
			raise Exception()

		# Check if folder name is already taken
		if self.name in os.listdir(os.path.normpath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir), "Archives"))):
			self.logger.error("Folder name already taken.")
			raise Exception()

		# Check iterative search arguments
		if self.search_type == "i" and not self.backward in [0, 1]:
			self.logger.error("Invalid backward argument. Type in '0' or '1'.")
			raise Exception()
		elif self.search_type != "s":
			self.logger.error("Invalid search type argument. Type in 's' or 'i'.")
			raise Exception()


	def execute(self):
		"""
		Create and send a query according to parsed arguments.
		"""
		self._check_arguments()
		q = Query(geojson=self.filename, date_range=(self.date_from, self.date_to), cloudcover_percentage=self.cloudcover_percentage, name=self.name)
		if self.search_type == 's':
			q.single_search()
		elif self.search_type == 'i':
			q.iter_search(day_step=self.day_step, iter_max=self.iter_max, backward=bool(self.backward))

###############
## Functions ##
###############

def main():
	parser = SentinelParser()
	try:
		parser.execute()
	except Exception as e:
		raise e

if __name__ == '__main__':
	main()
else:
	raise(Exception())