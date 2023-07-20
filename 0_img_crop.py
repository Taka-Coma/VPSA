# -*- coding: utf-8 -*-

from glob import glob
import cv2
import os

def main():
	datasets = ['roxford5k', 'rparis6k', 'revisitop1m']
	base_dir = './data/data/datasets/'

	for ds in datasets:
		os.makedirs(f'cropped_data/jpg/{ds}', exist_ok=True)

		#for img_path in glob(f'{base_dir}{ds}/jpg/*.jpg'):
		for img_path in glob(f'{base_dir}/{ds}/jpg/*/*.jpg'):
			img_name = img_path[img_path.rfind('/')+1:img_path.find('.jpg')]
			img_mid_path = img_path[img_path.find('jpg/')+1:img_path.rfind('/')]

			os.makedirs(f'cropped_data/jpg/{ds}/{img_mid_path}', exist_ok=True)

			img = cv2.imread(img_path)

			if img is None:
				print(img_path)
				continue

			x, y, _ = img.shape
			c_width = int(x/2)
			c_height = int(y/2)

			for s, yi in enumerate([0, 0.5, 1.0]):
				for t, xi in enumerate([0, 0.5, 1.0]):
					cropped = img[int(xi*c_width): int((xi+1)*c_width), int(yi*c_height): int((yi+1)*c_height)]
					cv2.imwrite(f'cropped_data/jpg/{ds}/{img_mid_path}/{img_name}_{s}{t}.jpg', cropped)

if __name__ == "__main__":
    main()
