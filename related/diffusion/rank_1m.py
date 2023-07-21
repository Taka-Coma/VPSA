#!/usr/bin/env python
# -*- coding: utf-8 -*-

" rank module "

import os
import time
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from dataset import Dataset
from knn import KNN
from diffusion import Diffusion
from sklearn import preprocessing
from evaluate import compute_map_and_print
import math
from glob import glob

def sigmoidWeight(sims):
	return sum([sim/(1+math.e**-sim) for sim in sims])

cache_dir_root = '/1m_ori'

def search(aggregation_method, gallery):
		n_query = len(queries)

		os.makedirs(args.cache_dir + f'{cache_dir_root}/ori', exist_ok=True)

		print(queries.shape)
		print(gallery.shape)

		start_time = time.time()
		scores = []

		current_len = gallery.shape[0]
		gallery_lst = [gallery]
		for path in glob('features/revisitop1m_ori/*'):
			head = path[path.rfind('/')+1:path.find('_')]

			tmp_gallery = np.nan_to_num(np.load(path))
			current_len += tmp_gallery.shape[0]
			gallery_lst.append(tmp_gallery)

			#print(current_len)

			if current_len < 10000000:
				continue

			gallery = np.vstack(gallery_lst)

			os.makedirs(args.cache_dir + f'{cache_dir_root}/{head}', exist_ok=True)

			print(queries.shape)
			print(gallery.shape)

			diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir + f'{cache_dir_root}/{head}')
			offline = diffusion.get_offline_results(args.truncation_size, args.kd)

			#offline = np.nan_to_num(offline)
			#features = preprocessing.normalize(offline, norm="l2", axis=1)
			features = offline

			tmp_scores = features[:n_query] @ features[n_query:].T
			tmp_scores = tmp_scores.toarray()

			for qid, row in enumerate(tmp_scores):
				if len(scores) < qid+1:
					scores.append([])

				if len(row) % 9 == 0:
					spl = len(row)//9
				else:
					spl = len(row)//10

				if aggregation_method == 'max':
					scores[qid] += [-max(grp) for grp in np.split(row, spl)]
				elif aggregation_method == 'avg':
					scores[qid] += [-np.mean(grp) for grp in np.split(row, spl)]
				elif aggregation_method == 'sigW':
					scores[qid] += [-sigmoidWeight(grp) for grp in np.split(row, spl)]
				else:
					print(aggregation_method, 'not supported')
					exit()

			gallery_lst = []
			current_len = 0


		gallery = np.vstack(gallery_lst)
		del gallery_lst

		os.makedirs(args.cache_dir + f'{cache_dir_root}/total', exist_ok=True)

		print(queries.shape)
		print(gallery.shape)

		diffusion = Diffusion(np.vstack([queries, gallery]), args.cache_dir + f'{cache_dir_root}/total')
		offline = diffusion.get_offline_results(args.truncation_size, args.kd)

		features = preprocessing.normalize(offline, norm="l2", axis=1)
		scores = features[:n_query] @ features[n_query:].T

		ranks = np.argsort(-scores.todense())

		print(ranks.shape)
		print(f'time passed: {time.time() - start_time}')

		evaluate(ranks)


def evaluate(ranks):
    gnd_name = os.path.splitext(os.path.basename(args.gnd_path))[0]
    with open(args.gnd_path, 'rb') as f:
        gnd = pickle.load(f)['gnd']
    compute_map_and_print(gnd_name.split("_")[-1], ranks.T, gnd)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_dir',
                        type=str,
                        default='./cache',
                        help="""
                        Directory to cache
                        """)
    parser.add_argument('--dataset_name',
                        type=str,
                        required=True,
                        help="""
                        Name of the dataset
                        """)
    parser.add_argument('--query_path',
                        type=str,
                        required=True,
                        help="""
                        Path to query features
                        """)
    parser.add_argument('--gallery_path',
                        type=str,
                        required=True,
                        help="""
                        Path to gallery features
                        """)
    parser.add_argument('--gnd_path',
                        type=str,
                        help="""
                        Path to ground-truth
                        """)
    parser.add_argument('-n', '--truncation_size',
                        type=int,
                        default=1000,
                        help="""
                        Number of images in the truncated gallery
                        """)
    parser.add_argument('--aggregation_method',
                        type=str,
                        default='max',
                        help="""
						Aggregation method of visual passages
                        """)
    args = parser.parse_args()
    args.kq, args.kd = 10, 50
    return args


if __name__ == "__main__":
    args = parse_args()
    if not os.path.isdir(args.cache_dir):
        os.makedirs(args.cache_dir)
    dataset = Dataset(args.query_path, args.gallery_path)
    queries, gallery = dataset.queries, dataset.gallery
    search(args.aggregation_method, gallery)

