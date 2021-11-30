# Experiments on ChestX-Ray14

This code is for reproducing fine-tuning ResNet-18 model on a data set from [CheXNet paper](https://arxiv.org/pdf/1711.05225) that predicted 14 common diagnoses using convolutional neural networks in over 100,000 NIH chest x-rays.

### NIH Dataset

To explore the full dataset, [download images from NIH (large, ~40gb compressed)](https://nihcc.app.box.com/v/ChestXray-NIHCC), extract all `tar.gz` files to a single folder and provide path as needed in code.

Use `batch_download_zips.py` to download the full dataset.

### Usage

Use `retrain.py` to fine-tune models on the full NIH data set.

```python
python retrain.py --reg_method None --reg_norm None --device 0

python retrain.py --reg_method constraint --reg_norm frob \
    --reg_extractor 5.728564437344309 --reg_predictor 2.5669480884876905 --scale_factor 1.0340072757925474 \
    --device 0
```

### Acknowledgments

Thanks to the authors of **[reproduce-chexnet](https://github.com/jrzech/reproduce-chexnet)** for providing their implementation publicly available. 