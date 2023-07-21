# -*- coding: utf-8 -*-

from glob import glob
import numpy as np

def main():
	x = [np.load(path) for path in glob('data/features/revisitop1m/*.npy')]
	X = np.vstack(x)

	print(X.shape)
	np.save('data/features/revisitop1m.npy', X)

if __name__ == "__main__":
    main()
