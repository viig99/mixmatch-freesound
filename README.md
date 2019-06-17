# FreeSound 2019 MixMatch + SpecAugment + Noisy
This repository tries to solve the multi-label freesound classification using [MixMatch: A Holistic Approach to Semi-Supervised Learning](https://arxiv.org/abs/1905.02249) and [SpecAugment](https://arxiv.org/abs/1904.08779).

Dataset is using the freesound - 2019 kaggle competion.

## Requirements
- Python 3.6+
- PyTorch 1.1
- **torchvision 0.3.0 (older versions are not compatible with this code)** 
- tensorboardX
- progress
- matplotlib
- numpy
- librosa

## Usage

### Train
Train the model by using the freesound 2019 curated and noisy data.

```
./train.sh
```

### Monitoring training progress
```
tensorboard --logdir=./result
```

## Results (Accuracy)
0.856 lwlwrap using 5% test set till now.

## References
```
@article{berthelot2019mixmatch,
  title={MixMatch: A Holistic Approach to Semi-Supervised Learning},
  author={Berthelot, David and Carlini, Nicholas and Goodfellow, Ian and Papernot, Nicolas and Oliver, Avital and Raffel, Colin},
  journal={arXiv preprint arXiv:1905.02249},
  year={2019}
}
```