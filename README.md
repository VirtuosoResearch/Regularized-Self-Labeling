### Overview

This repository provides the implementation for the paper "Improved Regularization and Robustness for Fine-tuning in Neural Networks", which will be presented as a poster paper in NeurIPS'21.

In this work, we propose a regularized self-labeling approach that combines regularization and self-training methods for improving the generalization and robustness properties of fine-tuning.  Our approach includes two components:

- First, we encode ***layer-wise regularization*** to penalize the model weights at different layers of the neural net.
- Second, we add ***self-labeling*** that relabels data points based on current neural net's belief and reweights data points whose confidence is low.

<div align=center><img src='./figures/main_figure.png' width="500"></div>

### Requirements

To install requirements:

```bash
pip install -r requirements.txt
```

### Data Preparation

We use seven image datasets in our paper. We list the link for downloading these datasets and describe how to prepare data to run our code below.

- [Aircrafts](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/): download and extract into `./data/aircrafts`
  - remove the class `257.clutter` out of the data directory
- [CUB-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html): download and extract into `./data/CUB_200_2011/`
- [Caltech-256](http://www.vision.caltech.edu/Image_Datasets/Caltech256/): download and extract into `./data/caltech256/`
- [Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html): download and extract into `./data/StanfordCars/`
- [Stanford-Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/): download and extract into `./data/StanfordDogs/`
- [Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/): download and extract into `./data/flowers/`
- [MIT-Indoor](http://web.mit.edu/torralba/www/indoor.html): download and extract into `./data/Indoor/`

Our code automatically handles the split of the datasets.

### Usage

Our algorithm (RegSL) interpolates between layer-wise regularization and self-labeling. Run the following commands for conducting experiments in this paper.

**Fine-tuning ResNet-101 on image classification tasks.** 

```python
python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_indoor.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.136809975858091 --reg_predictor 6.40780158171339 --scale_factor 2.52883770643206\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_aircrafts.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 1.18330556653284 --reg_predictor 5.27713618808711 --scale_factor 1.27679969876201\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_birds.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.204403908747731 --reg_predictor 23.7850606577679 --scale_factor 4.73803591794678\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_caltech.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.0867998872549272 --reg_predictor 9.4552942790218 --scale_factor 1.1785989596144\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_cars.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 1.3340347414257 --reg_predictor 8.26940794089601 --scale_factor 3.47676759842434\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_dogs.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.0561320847651626 --reg_predictor 4.46281825974388 --scale_factor 1.58722606909531\
    --device 1

python train_constraint.py --model ResNet101 \
    --config configs/config_constraint_flower.json \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.131991042311165 --reg_predictor 10.7674132173309 --scale_factor 4.98010215976503\
    --device 1
```

**Fine-tuning ResNet-18 under label noise.**

```Python
python train_label_noise.py --config configs/config_constraint_indoor.json --model ResNet18 \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 7.80246991703043 --reg_predictor 14.077402847906 \
    --noise_rate 0.2 --train_correct_label --reweight_epoch 5 --reweight_temp 2.0 --correct_epoch 10 --correct_thres 0.9 

python train_label_noise.py --config configs/config_constraint_indoor.json --model ResNet18 \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 8.47139398080791 --reg_predictor 19.0191127114923 \
    --noise_rate 0.4 --train_correct_label --reweight_epoch 5 --reweight_temp 2.0 --correct_epoch 10 --correct_thres 0.9 

python train_label_noise.py --config configs/config_constraint_indoor.json --model ResNet18 \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 10.7576018531961 --reg_predictor 19.8157649727473 \
    --noise_rate 0.6 --train_correct_label --reweight_epoch 5 --reweight_temp 2.0 --correct_epoch 10 --correct_thres 0.9 
    
python train_label_noise.py --config configs/config_constraint_indoor.json --model ResNet18 \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 9.2031662757248 --reg_predictor 6.41568500472423 \
    --noise_rate 0.8 --train_correct_label --reweight_epoch 5 --reweight_temp 1.5 --correct_epoch 10 --correct_thres 0.9 
```

**Fine-tuning Vision Transformer on noisy labels.** 

```Python
python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --reg_method none --reg_norm none \
    --lr 0.0001 --device 1 --noise_rate 0.4

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --reg_method none --reg_norm none \
    --lr 0.0001 --device 1 --noise_rate 0.8

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.7488074175044196 --reg_predictor 9.842955837419588 \
    --train_correct_label --reweight_epoch 24 --correct_epoch 18\
    --lr 0.0001 --device 1 --noise_rate 0.4

python train_label_noise.py --config configs/config_constraint_indoor.json \
    --model VisionTransformer --is_vit --img_size 224 --vit_type ViT-B_16 --vit_pretrained_dir pretrained/imagenet21k_ViT-B_16.npz \
    --reg_method constraint --reg_norm frob \
    --reg_extractor 0.1568903647089986 --reg_predictor 1.407080880079702 \
    --train_correct_label --reweight_epoch 18 --correct_epoch 2\
    --lr 0.0001 --device 1 --noise_rate 0.8
```

Please follow the instructions in [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) to download the pre-trained models. 

### Citation

If you find this repository useful, consider citing our work titled above.

### Acknowledgment

Thanks to the authors of the following repositories for providing their implementation publicly available.

- [mars-finetuning](https://github.com/henrygouk/mars-finetuning)
- [WS-DAN.PyTorch](https://github.com/GuYuc/WS-DAN.PyTorch)
- [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
