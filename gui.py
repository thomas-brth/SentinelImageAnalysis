# Sentinel-2 data extraction GUI

#############
## Imports ##
#############

## General imports
import os
import sys
from datetime import date

## GUI imports
import wx
from wx.adv import DatePickerCtrl
from wx.lib.intctrl import IntCtrl

## Custom imports
from image import Query, Image

###############
## Constants ##
###############

GENERAL_SIZE = (800, 600)
DEFAULT_POSITION = (200, 50)

#############
## Classes ##
#############

class SentinelApp(wx.App):
	"""
	"""
	def __init__(self, title : str = "Sentinel-2 image downloader", size : tuple = GENERAL_SIZE, position : tuple = DEFAULT_POSITION):
		super(SentinelApp, self).__init__()
		# Default frame style flag, with resizing and maximizing window disabled
		flags = wx.DEFAULT_FRAME_STYLE ^ (wx.RESIZE_BORDER | wx.MAXIMIZE_BOX)
		self.frame = SentinelFrame(
								  parent=None,
								  id=wx.ID_ANY,
								  title=title,
								  size=size,
								  position=position,
								  style=flags
								  )
		self.frame.Show(True)
		
class SentinelFrame(wx.Frame):
	"""
	"""
	def __init__(self, parent, id : int, title :str, size : tuple, position : tuple, style):
		super(SentinelFrame, self).__init__(parent=parent, id=id, title=title, size=size, pos=position, style=style)

		# Main panel with notebook
		self.main_panel = wx.Panel(parent=self, id=wx.ID_ANY, size=GENERAL_SIZE)
		self.notebook = wx.Notebook(parent=self.main_panel, style=wx.NB_TOP)

		# Query panel
		self.query_panel = QueryPanel(parent=self.notebook, size=GENERAL_SIZE)

		# Display panel
		self.display_panel = None

		# Notebook pages management
		self.notebook.AddPage(self.query_panel, "Query")
		self.notebook.AddPage(wx.Panel(parent=self.notebook, size=GENERAL_SIZE), "Display")

		# Sizer settings
		main_sizer = wx.BoxSizer(wx.VERTICAL)
		main_sizer.Add(self.notebook, 0, wx.EXPAND, 0)
		self.main_panel.SetSizer(main_sizer)

class QueryPanel(wx.Panel):
	"""
	"""
	def __init__(self, parent, size : tuple):
		super(QueryPanel, self).__init__(parent=parent, id=wx.ID_ANY, size=size)
		self.parent = parent
		width, height = size

		geojson_choices = os.listdir("GeoJSON")
		self.geojson_list = wx.ComboBox(parent=self, id=wx.ID_ANY, size=(3 / 8 * width, 1 / 12 * height), choices=geojson_choices, style=wx.CB_DROPDOWN | wx.CB_READONLY)

		# First panel
		panel_from = wx.Panel(parent=self, id=wx.ID_ANY, size=(width, 1 / 6 * height))
		label_from = wx.StaticText(parent=panel_from, id=wx.ID_ANY, label="From :")
		self.datectrl_from = DatePickerCtrl(parent=panel_from, id=wx.ID_ANY)

		sizer_from = wx.BoxSizer(wx.HORIZONTAL)
		sizer_from.Add(label_from, 0, wx.EXPAND | wx.ALL, 20)
		sizer_from.Add(self.datectrl_from, 0, wx.RIGHT | wx.TOP | wx.BOTTOM, 20)
		panel_from.SetSizer(sizer_from)

		# Second panel
		panel_to = wx.Panel(parent=self, id=wx.ID_ANY, size=(width, 1 / 6 * height))
		label_to = wx.StaticText(parent=panel_to, id=wx.ID_ANY, label="To :")
		self.datectrl_to = DatePickerCtrl(parent=panel_to, id=wx.ID_ANY)

		sizer_to = wx.BoxSizer(wx.HORIZONTAL)
		sizer_to.Add(label_to, 0, wx.EXPAND | wx.ALL, 20)
		sizer_to.Add(self.datectrl_to, 0, wx.RIGHT | wx.TOP | wx.BOTTOM, 20)
		panel_to.SetSizer(sizer_to)

		# Third panel
		panel_cc = wx.Panel(parent=self, id=wx.ID_ANY, size=(width, 1 / 6 * height))
		label_cc = wx.StaticText(parent=panel_cc, id=wx.ID_ANY, label="Cloud cover percentage :")
		self.intctrl_from = IntCtrl(parent=panel_cc, id=wx.ID_ANY)
		self.intctrl_to = IntCtrl(parent=panel_cc, id=wx.ID_ANY)

		sizer_cc = wx.BoxSizer(wx.HORIZONTAL)
		sizer_cc.Add(label_cc, 0, wx.EXPAND | wx.ALL, 20)
		sizer_cc.Add(self.intctrl_from, 0, wx.RIGHT | wx.TOP | wx.BOTTOM, 20)
		sizer_cc.Add(self.intctrl_to, 0, wx.RIGHT | wx.TOP | wx.BOTTOM, 20)
		panel_cc.SetSizer(sizer_cc)

		# Process button
		self.button = wx.Button(parent=self, id=wx.ID_ANY, size=(width * 1 / 6, height * 1 / 12), label="Process")
		self.button.Bind(wx.EVT_BUTTON, handler=self.process)

		# Main sizer
		main_sizer = wx.BoxSizer(wx.VERTICAL)
		main_sizer.Add(self.geojson_list, 0, wx.ALL | wx.CENTER, 20)
		main_sizer.Add(panel_from, 0, wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, 20)
		main_sizer.Add(panel_to, 0, wx.ALL | wx.EXPAND, 20)
		main_sizer.Add(panel_cc, 0, wx.RIGHT | wx.LEFT | wx.BOTTOM | wx.EXPAND, 20)
		main_sizer.Add(self.button, 0, wx.ALL | wx.CENTER, 20)

		self.SetSizer(main_sizer)

	def process(self, event):
		"""
		"""
		geojson = self.geojson_list.GetValue()
		date_from = self.datectrl_from.GetValue()
		date_to = self.datectrl_to.GetValue()
		cc_from = self.intctrl_from.GetValue()
		cc_to = self.intctrl_to.GetValue()
		q = Query(geojson, (date_from, date_to), cc_to)

		_bool = True
		while _bool:
			q.query_from_json()
			if q.length < 1:
				# No data message
				_bool = False
			else:
				name, cc = q.q_dataframe.head(1).index[0], q.q_dataframe.head(1)["cloudcoverpercentage"]
				_bool = False

class DisplayPanel(wx.Panel):
	"""
	"""
	def __init__(self, parent, size : tuple):
		super(DisplayPanel, self).__init__(parent=parent, id=wx.ID_ANY, size=size)
		self.parent = parent

###############
## Functions ##
###############

def main():
	app = SentinelApp()
	app.MainLoop()

if __name__ == '__main__':
	main()