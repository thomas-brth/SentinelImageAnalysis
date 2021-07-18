# Clustering Algorithms

#############
## Imports ##
#############

import numpy as np
import random as rd

###############
## Constants ##
###############

#############
## Classes ##
#############

class KMeansCluster():
	"""
	K-means clustering algorithm.

	Simple but can lead to unprecise clustering.
	"""
	def __init__(self, n_clusters : int = None, iter_max : int = 100):
		self.n_clusters = n_clusters
		self.iter_max = iter_max
		
		# Initialization
		self.k_means = {}
		
	def find_n_clusters(self):
		pass

	def initialize_means(self, data : np.ndarray):
		"""
		Initializes k-means values, with steady interval.
		"""
		for i in range (1, self.n_clusters + 1):
			self.k_means[i] = rd.uniform(data.min(), data.max())

	def sample_fit(self, data : np.ndarray, epsilon : float):
		"""
		Less precise fitness algorithm, but much quicker.
		"""
		sample = np.random.choice(data.reshape(1, -1)[0], size=10000)
		k_means, distances, labels = self.fit(sample, epsilon)
		distances, labels = self.compute_labels(data)
		return k_means, distances, labels

	def fit(self, data : np.ndarray, epsilon : float):
		"""
		Find k-means fitting data at best.
		"""
		delta = 100
		iter_count = 1
		self.initialize_means(data)

		data_shape = data.shape
		arr = data.reshape(1, -1)[0]

		while delta > epsilon and iter_count <= self.iter_max:
			print("=" * 40, flush=True)
			print(f"-- Loop n°{iter_count} --", flush=True)
			print(f"-- K-means : {self.k_means} --", flush=True)
			print(f"-- Delta : {delta} --", flush=True)
			print("=" * 40, flush=True)
			distances, labels = self.compute_labels(arr)
			old_means = self.update_means(distances, labels)

			delta = 0
			for k in range (1, self.n_clusters + 1):
				delta += abs(old_means[k] - self.k_means[k])
			delta /= self.n_clusters

			iter_count += 1

		return self.k_means, distances.reshape(data_shape), labels.reshape(data_shape)

	def compute_labels(self, data : np.ndarray):
		"""
		Compute the labels
		"""
		shape = data.shape
		if len(shape) > 1:
			data = data.reshape(1, -1)[0]

		n = data.shape[0]
		labels = np.full((n, ), -1)
		distances = np.full((n, ), 0.0)
		for i in range (n):
			dist_min, label = self.get_label(data[i])
			distances[i] = dist_min
			labels[i] = label
		return distances.reshape(shape), labels.reshape(shape)

	def get_label(self, val : float):
		"""
		Get the closest k-mean label to the given value.
		"""
		dist_min = np.sqrt((self.k_means[1] - val) ** 2)
		label = 1
		for k in range (2, self.n_clusters + 1):
			dist = np.sqrt((self.k_means[k] - val) ** 2)
			if dist < dist_min:
				dist_min = dist
				label = k
		return dist_min, label

	def update_means(self, distances : np.ndarray, labels : np.ndarray):
		"""
		Compute new means and update them.
		"""
		old_means = self.k_means.copy()
		delta = 0
		for k in range (1, self.n_clusters + 1):
			self.k_means[k] = distances[labels == k].mean()
		self.sort_means()

		return old_means

	def sort_means(self):
		"""
		Sort the different k-means values.
		"""
		l_means = [self.k_means[key] for key in self.k_means.keys()]
		l_means.sort()
		for i in range (1,self.n_clusters+1):
			self.k_means[i] = l_means[i-1]

###############
## Functions ##
###############

def main():
	pass

if __name__ == '__main__':
	main()
else:
	print(f"Module {__name__} imported.", flush=True)