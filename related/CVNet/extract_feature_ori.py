# written by Seongwon Lee (won4113@yonsei.ac.kr)

import os
import torch
import numpy as np
from glob import glob 

import core.checkpoint as checkpoint
from core.config import cfg
from model.CVNet_Rerank_model import CVNet_Rerank

from tqdm import tqdm
import torch
import torch.nn.functional as F
import test.test_loader_1m_ori as loader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

@torch.no_grad()
def setup_model():
    """Sets up a model for training or testing and log the results."""
    # Build the model
    print("=> creating CVNet_Rerank model")
    model = CVNet_Rerank(101, cfg.MODEL.HEADS.REDUCTION_DIM)
    print(model)
    cur_device = torch.cuda.current_device()
    model = model.cuda(device=cur_device)

    return model


@torch.no_grad()
def extract_feature(model, data_dir, scale_list):
    with torch.no_grad():
        test_loader = loader.construct_loader(data_dir, scale_list)

        img_feats = [[] for i in range(len(scale_list))] 

        for im_list in tqdm(test_loader):
            for idx in range(len(im_list)):
                im_list[idx] = im_list[idx].cuda()
                desc = model.extract_global_descriptor(im_list[idx])
                if len(desc.shape) == 1:
                    desc.unsqueeze_(0)
                desc = F.normalize(desc, p=2, dim=1)
                img_feats[idx].append(desc.detach().cpu())

        for idx in range(len(img_feats)):
            img_feats[idx] = torch.cat(img_feats[idx], dim=0)
            if len(img_feats[idx].shape) == 1:
                img_feats[idx].unsqueeze_(0)

        img_feats_agg = F.normalize(torch.mean(torch.cat([img_feat.unsqueeze(0) for img_feat in img_feats], dim=0), dim=0), p=2, dim=1)
        img_feats_agg = img_feats_agg.cpu().numpy()

    return img_feats_agg



def main():
		scale_list = [0.7071, 1.0, 1.4142]

		model = setup_model()
		checkpoint.load_checkpoint('pretrained/CVPR2022_CVNet_R101.pyth', model)

		torch.backends.cudnn.benchmark = False
		model.eval()

		print("extract database features")

		os.makedirs('data/features/revisitop1m_ori', exist_ok=True)

		### Extract features
		for data_dir in glob('data/datasets/revisitop1m/jpg/*'):
			dir_name = data_dir[data_dir.rfind('/')+1:]

			X = extract_feature(model, data_dir, scale_list)
			np.save(f'data/features/revisitop1m_ori/{dir_name}_cvnet_vp_glob.npy', X)

		torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
	main()
