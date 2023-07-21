# -*- coding: utf-8 -*-

import pickle

def main():
	for ds in ['roxford5k', 'rparis6k']:
		path = f'data/data/datasets/{ds}/gnd_{ds}.pkl'

		with open(path, 'rb') as f:
			data = pickle.load(f)

			imlist = []
			for im in data['imlist']:
				imlist.append(im)
				for i in range(3):
					for j in range(3):
						imlist.append(f'{im}_{i}{j}')

			data['imlist_vp'] = imlist

		with open(f'cropped_data/datasets/{ds}/gnd_{ds}.pkl', 'wb') as f:
			pickle.dump(data, f)

if __name__ == "__main__":
    main()
