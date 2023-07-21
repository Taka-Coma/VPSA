#!/usr/bin/env python
# -*- coding: utf-8 -*-

" knn module, all credits to faiss! "

import os
import numpy as np
import time
from tqdm import tqdm

import faiss


class BaseKNN(object):
    """KNN base class"""
    def __init__(self, database, method, mode=None):
        if database.dtype != np.float32:
            database = database.astype(np.float32)
        #if database.dtype != np.float16:
        #    database = database.astype(np.float16)
        self.N = len(database)
        self.D = database[0].shape[-1]
        self.database = database if database.flags['C_CONTIGUOUS'] \
                               else np.ascontiguousarray(database)
        self.mode = mode

    def add(self, batch_size=10000):
        """Add data into index"""
        if self.mode == 'nmslib':
            self.index.addDataPointBatch(self.database)
            self.index.createIndex({'post': 2}, print_progress=True)

        else:
            if self.N <= batch_size:
                self.index.add(self.database)
            else:
                [self.index.add(self.database[i:i+batch_size])
                        for i in tqdm(range(0, len(self.database), batch_size),
                                      desc='[index] add')]


    def search(self, queries, k):
        """Search
        Args:
            queries: query vectors
            k: get top-k results
        Returns:
            sims: similarities of k-NN
            ids: indexes of k-NN
        """
        if not queries.flags['C_CONTIGUOUS']:
            queries = np.ascontiguousarray(queries)
        if queries.dtype != np.float32:
            queries = queries.astype(np.float32)
        if self.mode == 'nmslib':
            ids, sims = self.index.knnQuery(queries, k)
        else:
            sims, ids = self.index.search(queries, k)
        return sims, ids


class KNN(BaseKNN):
    """KNN class
    Args:
        database: feature vectors in database
        method: distance metric
    """
    def __init__(self, database, method):
        super().__init__(database, method)
        self.index = {'cosine': faiss.IndexFlatIP,
                      'euclidean': faiss.IndexFlatL2}[method](self.D)
        self.index = faiss.IndexFlatIP(self.D)

        #self.index = faiss.index_factory(self.D, "PCA32,IVF100,Flat")
        #self.index.train(database)

        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            print('go to GPU')
            self.index = faiss.index_cpu_to_all_gpus(self.index)

        self.add()


class KNN_nmslib(BaseKNN):
	"""KNN class
		Args:
			database: feature vectors in database
			method: distance metric
	"""
	def __init__(self, database, method):
		super().__init__(database, method, 'nmslib')
		self.index =  nmslib.init(method='hnsw', space='cosinesimil')
		self.add()


class ANN(BaseKNN):
    """Approximate nearest neighbor search class
    Args:
        database: feature vectors in database
        method: distance metric
    """
    def __init__(self, database, method, M=128, nbits=8, nlist=316, nprobe=64):
        super().__init__(database, method)
        self.quantizer = {'cosine': faiss.IndexFlatIP,
                          'euclidean': faiss.IndexFlatL2}[method](self.D)
        self.index = faiss.IndexIVFPQ(self.quantizer, self.D, nlist, M, nbits)
        samples = database[np.random.permutation(np.arange(self.N))[:self.N // 5]]
        print("[ANN] train")
        self.index.train(samples)
        self.add()
        self.index.nprobe = nprobe

