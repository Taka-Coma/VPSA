# directory to data
DATA_DIR=./data
# directory to cache files
TMP_DIR=./tmp

# oxford5k, oxford105k, paris6k, paris106k
#DATASET=oxford5k
#DATASET=roxford5k
DATASET=rparis6k

# resnet or siamac
#FEATURE_TYPE=resnet
#FEATURE_TYPE=resnet_vp
#FEATURE_TYPE=resnet_vp_whole
#FEATURE_TYPE=cvnet
FEATURE_TYPE=cvnet_vp
#FEATURE_TYPE=cvnet_w_whole

TRUNC_SIZE=1000
#TRUNC_SIZE=5000
#TRUNC_SIZE=10000

#AGG_METHOD=avg
AGG_METHOD=max
#AGG_METHOD=sigW

#CUDA_VISIBLE_DEVICES=0,1
CUDA_VISIBLE_DEVICES=

.PHONY: rank
rank:
	python rank.py \
		--cache_dir $(TMP_DIR)/$(DATASET)_$(FEATURE_TYPE) \
		--query_path $(DATA_DIR)/query/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gallery_path $(DATA_DIR)/gallery/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gnd_path $(DATA_DIR)/gnd_$(DATASET).pkl \
		--dataset_name $(DATASET) \
		--truncation_size $(TRUNC_SIZE)


.PHONY: rank_1m
rank_1m:
	python rank_1m.py \
		--cache_dir $(TMP_DIR)/$(DATASET)_$(FEATURE_TYPE) \
		--query_path $(DATA_DIR)/query/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gallery_path $(DATA_DIR)/gallery/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gnd_path $(DATA_DIR)/gnd_$(DATASET).pkl \
		--dataset_name $(DATASET) \
		--truncation_size $(TRUNC_SIZE) \
		--aggregation $(AGG_METHOD)


.PHONY: rank_vp
rank_vp:
	python rank_vp.py \
		--cache_dir $(TMP_DIR)/$(DATASET)_$(FEATURE_TYPE) \
		--query_path $(DATA_DIR)/query/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gallery_path $(DATA_DIR)/gallery/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gnd_path $(DATA_DIR)/gnd_$(DATASET).pkl \
		--dataset_name $(DATASET) \
		--truncation_size $(TRUNC_SIZE) \
		--aggregation $(AGG_METHOD)

.PHONY: rank_vp_1m
rank_vp_1m:
	python rank_vp_1m.py \
		--cache_dir $(TMP_DIR)/$(DATASET)_$(FEATURE_TYPE) \
		--query_path $(DATA_DIR)/query/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gallery_path $(DATA_DIR)/gallery/$(DATASET)_$(FEATURE_TYPE)_glob.npy  \
		--gnd_path $(DATA_DIR)/gnd_$(DATASET).pkl \
		--dataset_name $(DATASET) \
		--truncation_size $(TRUNC_SIZE) \
		--aggregation $(AGG_METHOD)


.PHONY: mat2npy
mat2npy:
	python mat2npy.py \
		--dataset_name $(DATASET) \
		--feature_type $(FEATURE_TYPE) \
		--mat_dir $(DATA_DIR)


.PHONY: download
download:
	wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/oxford5k/gnd_oxford5k.pkl -O $(DATA_DIR)/gnd_oxford5k.pkl
	wget http://cmp.felk.cvut.cz/cnnimageretrieval/data/test/paris6k/gnd_paris6k.pkl -O $(DATA_DIR)/gnd_paris6k.pkl
	ln -s $(DATA_DIR)/gnd_oxford5k.pkl $(DATA_DIR)/gnd_oxford105k.pkl
	ln -s $(DATA_DIR)/gnd_paris6k.pkl $(DATA_DIR)/gnd_paris106k.pkl
	for dataset in oxford5k oxford105k paris6k paris106k; do \
		for feature in resnet; do \
			wget ftp://ftp.irisa.fr/local/texmex/corpus/diffusion/data/$$dataset\_$$feature.mat -O $(DATA_DIR)/$$dataset\_$$feature.mat; \
		done; \
	done
