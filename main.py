# Main script

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
import numpy as np
import time
import sys
sys.path.append("processing")

## Custom imports ##
from image import Image
from processing import mask, cluster, crop_2D, crop_3D, water_spectral_analysis, interpolation_2d, sharpen

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
	img = Image("S2A_MSIL2A_20210406T102021_N0300_R065_T32TLP_20210406T150850")

	plt.figure(1)
	ndvi = img.get_NDWIwb("R10m")

	plt.imshow(ndvi, cmap="ocean")
	plt.xticks([])
	plt.yticks([])
	
	plt.figure(2)
	scl = img.get_SCL("R10m")

	plt.imshow(scl, cmap="tab10")
	plt.xticks([])
	plt.yticks([])
	plt.show()

def test():
	img1 = Image("S2A_MSIL2A_20200222T163301_N0214_R083_T16RCV_20200222T205207")
	img2 = Image("S2A_MSIL2A_20200319T101021_N0214_R022_T32TQR_20200319T130518")

	bands1 = water_spectral_analysis(image=img1)
	bands2 = water_spectral_analysis(image=img2)

	x1 = np.array([bands1[band]['wl'] for band in bands1.keys()]) * 10 ** (-9)
	y1 = np.array([bands1[band]['mean'] for band in bands1.keys()])
	std1 = np.array([bands1[band]['std'] for band in bands1.keys()])
	y1_bis = y1 - 2 * std1
	y1_bis[y1_bis < 0] = 0

	x2 = np.array([bands2[band]['wl'] for band in bands2.keys()]) * 10 ** (-9)
	y2 = np.array([bands2[band]['mean'] for band in bands2.keys()])
	std2 = np.array([bands2[band]['std'] for band in bands2.keys()])
	y2_bis = y2 - 2 * std2
	y2_bis[y2_bis < 0] = 0

	plt.figure(1)
	lines1 = plt.semilogx(x1, y1, color='b', linestyle='-', linewidth=2)
	plt.semilogx(x1, y1 + 2 * std1, color='b', linestyle='--', linewidth=1)
	plt.semilogx(x1, y1_bis, color='b', linestyle='--', linewidth=1)
	lines2 = plt.semilogx(x2, y2, color='k', linestyle='-', linewidth=2)
	plt.semilogx(x2, y2 + 2 * std2, color='k', linestyle='--', linewidth=1)
	plt.semilogx(x2, y2_bis, color='k', linestyle='--', linewidth=1)
	plt.legend(lines1 + lines2, ['Image 1', 'Image 2'])

	plt.figure(2)
	plt.subplot(121)
	plt.imshow(img1.get_RGB(res="R10m", pLow=0.1, pHigh=0.75))
	plt.title('Image 1')
	plt.subplot(122)
	plt.imshow(img2.get_RGB(res="R10m", pLow=0.1, pHigh=0.75))
	plt.title('Image 2')
	plt.show()

def test_interp():
	img = Image("S2A_MSIL2A_20200222T163301_N0214_R083_T16RCV_20200222T205207")
	ndvi_10m = img.get_NDVI("R10m")
	ndvi_10m = crop_2D(ndvi_10m)
	ndvi_10m[~np.isfinite(ndvi_10m)] = -1
	ndvi_60m = img.get_NDVI("R60m")
	ndvi_60m = crop_2D(ndvi_60m)
	ndvi_60m[~np.isfinite(ndvi_60m)] = -1
	ndvi_interp = interpolation_2d(ndvi_60m, 6)
	ndvi_sharp = sharpen(ndvi_interp)

	plt.subplot(221)
	plt.imshow(ndvi_10m)
	plt.subplot(222)
	plt.imshow(ndvi_60m)
	plt.subplot(223)
	plt.imshow(ndvi_interp)
	plt.subplot(224)
	plt.imshow(ndvi_sharp)
	plt.show()

if __name__ == '__main__':
	main()