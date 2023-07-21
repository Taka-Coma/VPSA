# -*- coding: utf-8 -*-

import pickle
import numpy as np

def main():
	dataset = 'roxford5k'
	#dataset = 'rparis6k'
	#with open(f'cropped_data/features/{dataset}_w_whole.pkl', 'rb') as f:
	#with open(f'cropped_data/features/{dataset}_vp.pkl', 'rb') as f:
	#with open(f'cropped_data/features/{dataset}_ori.pkl', 'rb') as f:

	with open(f'data/data/features/cvnet_Q_{dataset}.npy', 'rb') as f:
		mat = pickle.load(f)
		print(mat.shape)
		np.save(f'data/data/features/query/{dataset}_cvnet_glob.npy', mat)

	with open(f'data/data/features/cvnet_X_{dataset}.npy', 'rb') as f:
		mat = pickle.load(f)
		print(mat.shape)
		np.save(f'data/data/features/gallery/{dataset}_cvnet_glob.npy', mat)

	#print(mat['Q'].T.shape)
	#print(mat['X'].T.shape)

	#np.save(f'data/data/features/query/{dataset}_resnet_vp_whole_glob.npy', mat['Q'].T)
	#np.save(f'data/data/features/gallery/{dataset}_resnet_vp_whole_glob.npy', mat['X'].T)
	#np.save(f'data/data/features/query/{dataset}_resnet_vp_glob.npy', mat['Q'].T)
	#np.save(f'data/data/features/gallery/{dataset}_resnet_vp_glob.npy', mat['X'].T)
	#np.save(f'data/data/features/query/{dataset}_resnet_glob.npy', mat['Q'].T)
	#np.save(f'data/data/features/gallery/{dataset}_resnet_glob.npy', mat['X'].T)

if __name__ == "__main__":
    main()
