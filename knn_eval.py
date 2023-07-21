# EXAMPLE_EVALUATE  Code to evaluate example results on ROxford and RParis datasets.
# Revisited protocol has 3 difficulty setups: Easy (E), Medium (M), and Hard (H), 
# and evaluates the performance using mean average precision (mAP), as well as mean precision @ k (mP@k)
#
# More details about the revisited annotation and evaluation can be found in:
# Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., Revisiting Oxford and Paris: Large-Scale Image Retrieval Benchmarking, CVPR 2018
#
# Authors: Radenovic F., Iscen A., Tolias G., Avrithis Y., Chum O., 2018

import os
import numpy as np
import math

import pickle

from dataset import configdataset
from evaluate import compute_map

import time

#---------------------------------------------------------------------
# Set data folder and testing parameters
#---------------------------------------------------------------------
# Set data folder, change if you have downloaded the data somewhere else
data_root = 'data/data/'

# Set test dataset: roxford5k | rparis6k
test_dataset = 'roxford5k'
#test_dataset = 'rparis6k'

aggregation_method='max'
#aggregation_method='avg'
#aggregation_method='sigW'

def sigmoidWeight(sims):
	return sum([sim/(1+math.e**-sim) for sim in sims])

#---------------------------------------------------------------------
# Evaluate
#---------------------------------------------------------------------

print('>> {}: Evaluating test dataset...'.format(test_dataset)) 
# config file for the dataset
# separates query image list from database image list, when revisited protocol used
cfg = configdataset(test_dataset, os.path.join(data_root, 'datasets'))

# load query and database features
print('>> {}: Loading features...'.format(test_dataset))    
#with open(f'cropped_data/features/{test_dataset}_ori.pkl', 'rb') as f:
#with open(f'cropped_data/features/{test_dataset}_vp.pkl', 'rb') as f:
with open(f'cropped_data/features/{test_dataset}_w_whole.pkl', 'rb') as f:
	features = pickle.load(f)
Q = features['Q']
X = features['X']

#Q = np.load(f'data/data/features/query/{test_dataset}_cvnet_vp_glob.npy').T
#X = np.load(f'data/data/features/gallery/{test_dataset}_cvnet_vp_glob.npy').T

start_time = time.time()
# perform search
print('>> {}: Retrieval...'.format(test_dataset))
sim = np.dot(X.T, Q)

print(sim.shape)

scores = []
for row in sim.T:
	#spl = len(row)//1
	if len(row) % 9 ==0:
		spl = len(row)//9
	else:
		spl = len(row)//10

	if aggregation_method == 'max':
		scores.append([-max(grp) for grp in np.split(row, spl)])
	elif aggregation_method == 'avg':
		scores.append([-np.mean(grp) for grp in np.split(row, spl)])
	elif aggregation_method == 'sigW':
		scores.append([-sigmoidWeight(grp) for grp in np.split(row, spl)])
	else:
		print(aggregation_method, 'not supported')
		exit()

ranks = np.argsort(scores, axis=1).T

print(f'time passed: {time.time() - start_time}')

print(ranks.shape)

print(aggregation_method)

# revisited evaluation
gnd = cfg['gnd']

# evaluate ranks
ks = [1, 5, 10]

# search for easy
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['hard']])
    gnd_t.append(g)
mapE, apsE, mprE, prsE = compute_map(ranks, gnd_t, ks)

# search for easy & hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk']])
    gnd_t.append(g)
mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

# search for hard
gnd_t = []
for i in range(len(gnd)):
    g = {}
    g['ok'] = np.concatenate([gnd[i]['hard']])
    g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
    gnd_t.append(g)
mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

print('>> {}: mAP E: {}, M: {}, H: {}'.format(test_dataset, np.around(mapE*100, decimals=2), np.around(mapM*100, decimals=2), np.around(mapH*100, decimals=2)))
print('>> {}: mP@k{} E: {}, M: {}, H: {}'.format(test_dataset, np.array(ks), np.around(mprE*100, decimals=2), np.around(mprM*100, decimals=2), np.around(mprH*100, decimals=2)))
