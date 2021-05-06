# Main script

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
import numpy as np
import time

## Custom imports ##
from image import Image
from processing import mask, cluster, crop_2D, crop_3D

###############
## Constants ##
###############

Fx = np.array([
			   [-1, 0, 1],
			   [-2, 0, 2],
			   [-1, 0, 1]
			   ])
Fy = np.array([
			   [-1, -2, -1],
			   [0, 0, 0],
			   [1, 2, 1]
			   ])
#############
## Classes ##
#############

###############
## Functions ##
###############

def corr(band, corr_coef):
	arr = band - band * corr_coef
	arr[arr < 0] = 0
	return arr

def water():
	img = Image("S2B_MSIL2A_20200328T095029_N0214_R079_T33TXM_20200328T122832")
	w_mask = mask.water_mask(img, "R20m")
	rgb = img.get_RGB(res="R20m", pLow=0.1, pHigh=0.5)

	b5 = img.load_single_band("R20m", "B05") / 10000 * 3.5
	b3 = img.load_single_band("R20m", "B03") / 10000 * 2.5
	index = b5/b3
	index[~w_mask] = np.nan

	plt.subplot(121)
	plt.imshow(rgb)
	plt.imshow(index, cmap="magma")
	plt.xticks([])
	plt.yticks([])
	plt.title("True Color Image")
	
	plt.subplot(122)
	b4 = img.load_single_band("R10m", "B04") / 10000 *2.9
	b3 = img.load_single_band("R10m", "B03") / 10000 *3.1
	b2 = img.load_single_band("R10m", "B02") / 10000 *3.0
	
	cor = 0.036
	b4 = corr(b4, cor)
	b3 = corr(b3, 2 * cor)
	b2 = corr(b2, 3.25 * cor)

	b4[b4 > 1] = 1
	b3[b3 > 1] = 1
	b2[b2 > 1] = 1

	rgb = np.dstack((b4, b3, b2))
	plt.imshow(rgb)
	plt.xticks([])
	plt.yticks([])
	plt.title("Enhanced True Color Image")
	plt.show()

def filter():
	img = Image("S2A_MSIL2A_20210425T105021_N0300_R051_T31UCP_20210425T135357")

	b4 = img.load_single_band("R10m", "B04") / 10000 *2.9
	b3 = img.load_single_band("R10m", "B03") / 10000 *3.1
	b2 = img.load_single_band("R10m", "B02") / 10000 *3.0
	
	cor = 0.036
	b4 = corr(b4, cor)
	b3 = corr(b3, 2 * cor)
	b2 = corr(b2, 3.25 * cor)

	b4[b4 > 1] = 1
	b3[b3 > 1] = 1
	b2[b2 > 1] = 1

	gray = 0.2989 * b4 + 0.5870 * b3 + 0.1140 * b2
	#gray = gray[750:1006, 250:506]

	n, m = gray.shape
	res1 = np.zeros((n, m))
	res2 = np.zeros((n, m))
	for i in range(1, n-1):
		for j in range(1, m-1):
			res1[i, j] = np.sum(gray[i-1:i+2, j-1:j+2] * Fx)
			res2[i, j] = np.sum(gray[i-1:i+2, j-1:j+2] * Fy)
	res1[res1 > 1] = 1
	res1[res1 < 0] = 0
	res2[res2 > 1] = 1
	res2[res2 < 0] = 0

	plt.figure(1)
	plt.imshow(gray)
	plt.figure(2)
	plt.subplot(121)
	plt.imshow(res1)
	plt.subplot(122)
	plt.imshow(res2)
	plt.show()

def main():
	img = Image("S2A_MSIL2A_20200206T111231_N0214_R137_T30TXT_20200206T122704")

	plt.figure(1)
	ndvi = img.get_NDWIwb("R10m")
	ndvi = crop_2D(ndvi)

	plt.imshow(ndvi, cmap="ocean")
	plt.xticks([])
	plt.yticks([])

	plt.figure(2)
	b4 = img.load_single_band("R10m", "B04") / 10000 *2.9
	b3 = img.load_single_band("R10m", "B03") / 10000 *3.1
	b2 = img.load_single_band("R10m", "B02") / 10000 *3.0
	
	cor = 0.036
	b4 = corr(b4, cor)
	b3 = corr(b3, 2 * cor)
	b2 = corr(b2, 3.25 * cor)

	b4[b4 > 1] = 1
	b3[b3 > 1] = 1
	b2[b2 > 1] = 1

	rgb = np.dstack([b4, b3, b2])
	rgb = crop_3D(rgb)
	plt.imshow(rgb)
	plt.xticks([])
	plt.yticks([])
	plt.show()

if __name__ == '__main__':
	main()