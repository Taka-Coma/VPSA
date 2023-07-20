# -*- coding: utf-8 -*-

from torch.utils.model_zoo import load_url
from cirtorch.networks.imageretrievalnet import init_network, extract_vectors
from cirtorch.utils.whiten import whitenapply
from torchvision import transforms
import pickle
import os

pretrained = 'http://cmp.felk.cvut.cz/cnnimageretrieval/data/networks/retrieval-SfM-120k/rSfM120k-tl-resnet101-gem-w-a155e54.pth'
whitening = 'retrieval-SfM-120k'
image_size = 1024
multiscale = '[1, 1/2**(1/2), 1/2]'
gpu_id = '0'

def main():
	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
	for ds in ['roxford5k', 'rparis6k']:
		with open(f'data/data/datasets/{ds}/gnd_{ds}.pkl', 'rb') as f:
			data = pickle.load(f)

		qimg = [f'data/data/datasets/{ds}/jpg/{n}.jpg' for n in data['qimlist']]
		bbox = [tuple(row['bbx']) for row in data['gnd']]

		dimg = []
		for n in data['imlist']:
			## Visual passage
			for i in range(3):
				for j in range(3):
					dimg.append(f'cropped_data/jpg/{ds}/{n}_{i}{j}.jpg')

			## with whole
			#dimg.append(f'data/data/datasets/{ds}/jpg/{n}.jpg')

		qvecs = calc_vectors(qimg, bbox)
		dvecs = calc_vectors(dimg)

		with open(f'cropped_data/features/{ds}_vp.pkl', 'wb') as f:
		#with open(f'cropped_data/features/{ds}_w_whole.pkl', 'wb') as f:
		#with open(f'cropped_data/features/{ds}_ori.pkl', 'wb') as f:
			pickle.dump({'Q': qvecs, 'X':dvecs}, f)


def calc_vectors(images, bbox=None):
	state = load_url(pretrained, model_dir='models/')

	net_params = {}
	net_params['architecture'] = state['meta']['architecture']
	net_params['pooling'] = state['meta']['pooling']
	net_params['local_whitening'] = state['meta'].get('local_whitening', False)
	net_params['regional'] = state['meta'].get('regional', False)
	net_params['whitening'] = state['meta'].get('whitening', True)
	net_params['mean'] = state['meta']['mean']
	net_params['std'] = state['meta']['std']
	net_params['pretrained'] = False

	net = init_network(net_params)
	net.load_state_dict(state['state_dict'])

	ms = list(eval(multiscale))

	net.cuda()
	net.eval()

	normalize = transforms.Normalize(
					mean=net.meta['mean'],
					std=net.meta['std']
					)
	transform = transforms.Compose([
					transforms.ToTensor(),
					normalize
	])

	if bbox:
		vecs = extract_vectors(net, images, image_size, transform, bbxs=bbox, ms=ms)
	else:
		vecs = extract_vectors(net, images, image_size, transform, ms=ms)

	return vecs.numpy()

if __name__ == "__main__":
    main()
