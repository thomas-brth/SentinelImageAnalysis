# Test script

#############
## Imports ##
#############

## General imports ##
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image as Img
import numpy as np
import time
import sys
sys.path.append("utils")
sys.path.append("utils\\processing")

## Custom imports ##
from sentIA import *

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

def main():
	img = Image("S2A_MSIL2A_20210406T102021_N0300_R065_T32TLP_20210406T150850")

	plt.figure(1)
	rgb = img.get_RGB(res="R10m", pLow=0.01, pHigh=0.75)
	ndwi = img.get_NDWIwb(res="R10m")
	k_means = cluster.KMeansCluster(n_clusters=2, iter_max=1000)
	k_means.sample_fit(data=ndwi, epsilon=0.001)
	dist, labels = k_means.compute_labels(data=ndwi)

	plt.subplot(121)
	plt.imshow(rgb)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.imshow(labels, cmap="tab10")
	plt.xticks([])
	plt.yticks([])

	plt.show()

def test_spectral():
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

def lytton_analysis():
	# 01/06 Image
	img1 = Image("S2A_MSIL2A_20210601T185921_N0300_R013_T10UEA_20210601T232218")
	b8_1 = img1.load_single_band("R10m", "B08") / 10000 * 2.3
	b4_1 = img1.load_single_band("R10m", "B04") / 10000 * 2.9
	b3_1 = img1.load_single_band("R10m", "B03") / 10000 * 3.1
	b2_1 = img1.load_single_band("R10m", "B02") / 10000 * 3.0

	# 29/06 Image
	img2 = Image("S2B_MSIL2A_20210629T190919_N0300_R056_T10UEA_20210629T215319")
	b8_2 = img2.load_single_band("R10m", "B08") / 10000 * 2.3
	b4_2 = img2.load_single_band("R10m", "B04") / 10000 * 2.9
	b3_2 = img2.load_single_band("R10m", "B03") / 10000 * 3.1
	b2_2 = img2.load_single_band("R10m", "B02") / 10000 * 3.0

	# Norms
	ndvi_norm = MidpointNormalize(vmin=0, vmax=1, midpoint=0.5)
	ndwi_norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

	# RGB Figure
	plt.figure(1)
	plt.suptitle("RGB image (B4, B3, B2)")
	plt.subplot(121)
	plt.title("01/06/2021")
	plt.imshow(np.dstack([b4_1, b3_1, b2_1]))
	plt.xticks([])
	plt.yticks([])

	plt.subplot(122)
	plt.title("29/06/2021")
	plt.imshow(np.dstack([b4_2, b3_2, b2_2]))
	plt.xticks([])
	plt.yticks([])
	# FIR Figure
	plt.figure(2)
	plt.suptitle("FIR image (B8, B4, B3)")
	plt.subplot(121)
	plt.title("01/06/2021")
	plt.imshow(np.dstack([b8_1, b4_1, b3_1]))
	plt.xticks([])
	plt.yticks([])

	plt.subplot(122)
	plt.title("29/06/2021")
	plt.imshow(np.dstack([b8_2, b4_2, b3_2]))
	plt.xticks([])
	plt.yticks([])

	# NDVI Figure
	plt.figure(3)
	plt.suptitle("NDVI (water & snow mask applied)")
	plt.subplot(121)
	plt.title("01/06/2021")
	ndvi1 = img1.get_NDVI("R10m")
	wm = mask.water_mask(img1, "R10m", threshold=0.02)
	ndvi1[wm] = np.nan
	plt.imshow(ndvi1, cmap="RdYlGn", norm=ndvi_norm)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title("29/06/2021")
	ndvi2 = img2.get_NDVI("R10m")
	wm = mask.water_mask(img2, "R10m", threshold=0.02)
	ndvi2[wm] = np.nan
	plt.imshow(ndvi2, cmap="RdYlGn", norm=ndvi_norm)
	plt.xticks([])
	plt.yticks([])

	# NDWI Figure
	plt.figure(4)
	plt.suptitle("NDMI (Normalized Difference Moisture Index, water & snow mask applied)")
	plt.subplot(121)
	plt.title("01/06/2021")
	ndwi1 = img1.get_NDWIveg("R20m")
	wm = mask.water_mask(img1, "R20m", threshold=0.02)
	ndwi1[wm] = np.nan
	plt.imshow(ndwi1, cmap="jet_r", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title("29/06/2021")
	ndwi2 = img2.get_NDWIveg("R20m")
	wm = mask.water_mask(img2, "R20m", threshold=0.02)
	ndwi2[wm] = np.nan
	plt.imshow(ndwi2, cmap="jet_r", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])

	# NDVI pixels distribution
	plt.figure(5)
	plt.suptitle("NDMI pixels distribution")
	s1 = ndwi1.reshape(1, -1)[0]
	s2 = ndwi2.reshape(1, -1)[0]
	plt.hist(s1, bins=500, histtype='step', label="01/06/2021")
	plt.hist(s2, bins=500, histtype='step', label="29/06/2021")
	plt.xlabel("NDMI")
	plt.ylabel("Number of pixels")
	plt.legend()

	plt.show()

def lytton_fire():
	img = Image("Milas_Fire_1")
	b8 = img.load_single_band("R10m", "B08") / 10000 * 2.3
	b4 = img.load_single_band("R10m", "B04") / 10000 * 2.9
	b3 = img.load_single_band("R10m", "B03") / 10000 * 3.1
	b2 = img.load_single_band("R10m", "B02") / 10000 * 3.0
	rgb = np.dstack((b4, b3, b2))
	fir = np.dstack((b8, b4, b3))
	norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

	plt.figure(1)
	plt.title("SWIR1 image (B11, B8A, B2)")
	plt.text(20, 20, "Lytton, 04/07/2021, Sentinel-2 L2A Image", color="white")
	plt.imshow(img.get_Agriculture("R20m"))
	plt.xticks([])
	plt.yticks([])

	plt.figure(2)
	plt.title("FIR image (B8, B4, B3)")
	plt.text(20, 20, "Lytton, 04/07/2021, Sentinel-2 L2A Image", color="white")
	plt.imshow(fir)
	plt.xticks([])
	plt.yticks([])

	plt.figure(3)
	plt.title("NDVI")
	plt.text(20, 20, "Lytton, 04/07/2021, Sentinel-2 L2A Image", color="black")
	ndvi = img.get_NDVI("R10m")
	plt.imshow(ndvi, cmap="RdYlGn")
	plt.xticks([])
	plt.yticks([])
	plt.colorbar(norm=norm)

	plt.figure(4)
	plt.title("NDMI")
	plt.text(20, 20, "Lytton, 04/07/2021, Sentinel-2 L2A Image", color="black")
	ndmi = img.get_NDWI("R20m")
	plt.imshow(ndmi, cmap="jet_r")
	plt.xticks([])
	plt.yticks([])
	plt.colorbar(norm=norm)

	plt.figure(5)
	plt.title("BAI")
	plt.text(20, 20, "Lytton, 04/07/2021, Sentinel-2 L2A Image", color="black")
	bai = (b8 - b3) / (b8 + b3)
	plt.imshow(bai, cmap="RdYlGn")
	plt.xticks([])
	plt.yticks([])
	plt.colorbar(norm=norm)

	plt.show()

def gonfaron_analysis():
	# 26/07 Image
	img1 = Image("Gonfaron_Fire_0")
	img1.pproc['pHigh'] = 0.5

	# 07/08 Image
	img2 = Image("Gonfaron_Fire_2")
	img2.pproc['pHigh'] = 0.5

	# Norm
	ndvi_norm = MidpointNormalize(vmin=-0.5, vmax=1, midpoint=0.25)
	ndwi_norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

	# RGB Figure
	plt.figure(1)
	plt.suptitle("RGB image (B4, B3, B2)", fontsize=24, fontweight='bold')
	plt.subplot(121)
	plt.title("02/08/2021", fontsize=16)
	im = plt.imshow(img1.get_RGB('R10m'))
	plt.xticks([])
	plt.yticks([])

	plt.subplot(122)
	plt.title("22/08/2021", fontsize=16)
	plt.imshow(img2.get_RGB('R10m'))
	plt.xticks([])
	plt.yticks([])
	
	# FIR Figure
	plt.figure(2)
	plt.suptitle("FIR image (B8, B4, B3)", fontsize=24, fontweight='bold')
	plt.subplot(121)
	plt.title("02/08/2021", fontsize=16)
	plt.imshow(img1.get_FIR('R10m'))
	plt.xticks([])
	plt.yticks([])

	plt.subplot(122)
	plt.title("22/08/2021", fontsize=16)
	plt.imshow(img2.get_FIR('R10m'))
	plt.xticks([])
	plt.yticks([])

	# NDVI Figure
	plt.figure(3)
	plt.suptitle("Normalized Diffirence vegetation Index", fontsize=24, fontweight='bold')
	plt.subplot(121)
	plt.title("02/08/2021", fontsize=16)
	ndvi1 = img1.get_NDVI("R10m")
	plt.imshow(ndvi1, cmap="cubehelix_r", norm=ndvi_norm)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title("22/08/2021", fontsize=16)
	ndvi2 = img2.get_NDVI("R10m")
	plt.imshow(ndvi2, cmap="cubehelix_r", norm=ndvi_norm)
	plt.xticks([])
	plt.yticks([])

	# NDWI Figure
	plt.figure(4)
	plt.suptitle("Normalized Diffirence Water Index", fontsize=24, fontweight='bold')
	plt.subplot(121)
	plt.title("02/08/2021", fontsize=16)
	ndwi1 = img1.get_NDWI("R10m")
	plt.imshow(ndwi1, cmap="viridis", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title("22/08/2021", fontsize=16)
	ndwi2 = img2.get_NDWI("R10m")
	plt.imshow(ndwi2, cmap="viridis", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])

	# NDMI Figure
	plt.figure(5)
	plt.suptitle("Normalized Diffirence Moisture Index", fontsize=24, fontweight='bold')
	plt.subplot(121)
	plt.title("02/08/2021", fontsize=16)
	ndmi1 = img1.get_NDMI("R20m")
	count = np.count_nonzero(~np.isnan(ndmi1))
	m = scl_based_mask(img2, 'R20m')
	ndmi1[m] = np.nan
	plt.imshow(ndmi1, cmap="magma_r", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])
	
	plt.subplot(122)
	plt.title("22/08/2021", fontsize=16)
	ndmi2 = img2.get_NDMI("R20m")
	ndmi2[m] = np.nan
	plt.imshow(ndmi2, cmap="magma_r", norm=ndwi_norm)
	plt.xticks([])
	plt.yticks([])

	# NDMI difference Figure
	plt.figure(6)
	plt.suptitle("NDMI Difference", fontsize=24, fontweight='bold')
	diff = abs(ndmi1 - ndmi2)
	plt.imshow(diff)
	plt.colorbar()
	plt.xticks([])
	plt.yticks([])

	# NDMI difference mask Figure
	plt.figure(7)
	plt.suptitle("NDMI Difference mask", fontsize=24, fontweight='bold')
	m_diff = diff > 0.35
	plt.imshow(m_diff)
	plt.xticks([])
	plt.yticks([])

	## Area covered by forest fire ##
	relative_area = np.count_nonzero(m_diff) / count
	print(f"Image area is about {img1.total_area} km2")
	print(f"Burnt area is about {relative_area * img1.total_area} km2")
	## About 49.02948925580708 km2 burnt

	# NDMI pixels distribution
	plt.figure(8)
	plt.suptitle("NDMI pixels distribution", fontsize=24, fontweight='bold')
	s1 = ndmi1.reshape(1, -1)[0]
	s2 = ndmi2.reshape(1, -1)[0]
	plt.hist(s1, bins=500, histtype='step', label="02/08/2021")
	plt.hist(s2, bins=500, histtype='step', label="22/08/2021")
	plt.xlabel("NDMI")
	plt.ylabel("Number of pixels")
	plt.legend()

	plt.show()

def gonfaron_animation():
	plt.rcParams['axes.facecolor'] = 'black'

	# 02/08 Image
	img1 = Image("Gonfaron_Fire_0")
	img1.pproc['pHigh'] = 1

	# 17/08 Image
	img2 = Image("Gonfaron_Fire_1")
	img2.pproc['pHigh'] = 1

	# 22/08 Image
	img3 = Image("Gonfaron_Fire_2")
	img3.pproc['pHigh'] = 1

	# Norm
	ndvi_norm = MidpointNormalize(vmin=-0.5, vmax=1, midpoint=0.25)
	ndwi_norm = MidpointNormalize(vmin=-1, vmax=0.2, midpoint=-0.4)
	ndmi_norm = MidpointNormalize(vmin=-1, vmax=1, midpoint=0)

	arr1 = img1.get_NDVI('R10m')
	arr2 = img2.get_NDVI('R10m')
	arr3 = img3.get_NDVI('R10m')

	l = [arr1, arr2, arr3]
	dates = ["02/08/2021", "17/08/2021", "22/08/2021"]

	fig = plt.figure(figsize=(9, 8))
	fig.patch.set_facecolor('black')
	im = plt.imshow(arr1, cmap='cubehelix_r', norm=ndvi_norm)
	im.axes.spines['top'].set_color('white')
	im.axes.spines['left'].set_color('white')
	im.axes.spines['right'].set_color('white')
	im.axes.spines['bottom'].set_color('white')
	plt.xticks([])
	plt.yticks([])
	plt.suptitle("Normalized Difference Vegetation Index", fontsize=24, fontweight='bold', color='white')
	plt.title(dates[0], fontsize=16, color='white')
	col = plt.colorbar(fraction=0.058, pad=0.025, orientation='horizontal')
	col.ax.tick_params(color='white', labelcolor='white')
	col.outline.set_edgecolor('white')
	def update(i):
		im.set_array(l[i])
		im.axes.set_title(dates[i])
		return [im]

	anim = FuncAnimation(fig, update, frames=3, interval=1500)
	anim.save("Images//NDVI_anim.gif", writer='imagemagick', savefig_kwargs={'facecolor':'black'})
	#plt.show()

if __name__ == '__main__':
	gonfaron_analysis()