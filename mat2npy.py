# -*- coding: utf-8 -*-

from scipy.io import loadmat
import numpy as np

def main():
	#dataset = 'roxford5k'
	dataset = 'rparis6k'
	mat = loadmat(f'data/data/features/{dataset}_resnet_rsfm120k_gem.mat')

	print(mat['Q'].T.shape)
	print(mat['X'].T.shape)

	np.save(f'data/data/features/query/{dataset}_resnet_glob.npy', mat['Q'].T)
	np.save(f'data/data/features/gallery/{dataset}_resnet_glob.npy', mat['X'].T)

if __name__ == "__main__":
    main()
