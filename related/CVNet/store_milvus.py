# -*- coding: utf-8 -*-

from glob import glob
import numpy as np

from pymilvus import (
    connections,
	utility,
	FieldSchema,
	CollectionSchema,
	DataType,
	Collection,
	)


def main():
	connections.connect('default', host='db', port='19530')

	schema = CollectionSchema([
		FieldSchema(
			name = 'vp_id',
			dtype = DataType.INT64,
			is_primary = True
		),
		FieldSchema(
			name = 'embeddings',
			dtype = DataType.FLOAT_VECTOR,
			dim = 2048
		)
	])


	#cols = ['roxford5k', 'rparis6k']
	cols = ['rparis6k']

	for col in cols:
		print(f'Collection: {col}')

		collection = Collection(
			name = f'{col}1m',
			schema = schema,
			using = 'default'
		)

		embs_all = np.load(f'data/features/gallery/{col}_cvnet_vp_glob.npy')

		last_vp_id = 0
		for embs in np.array_split(embs_all, 7):
			vp_ids = list(range(last_vp_id, last_vp_id+embs.shape[0]))
			entities = [vp_ids, embs]
			collection.insert(entities)

			last_vp_id = vp_ids[-1]+1

		print(f'Finished: {last_vp_id}')
		for path in glob('data/features/revisitop1m/*.npy'):
			embs = np.load(path)
			vp_ids = list(range(last_vp_id, last_vp_id+embs.shape[0]))
			entities = [vp_ids, embs]
			collection.insert(entities)

			last_vp_id = vp_ids[-1]+1

			print(f'Finished: {last_vp_id}')

		collection.flush()

		index = {
			'index_type': 'IVF_FLAT',
			'metric_type': 'IP',
			'params': {'nlist': 128}
		}

		collection.create_index('embeddings', index)

if __name__ == "__main__":
    main()
