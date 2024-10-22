# Plug-and-Play Monte Carlo

This is the code repo for the paper **"Provable Probabilistic Imaging using Score-based Generative Priors"** ([**IEEE TCI**](https://ieeexplore.ieee.org/document/10645293) | [**arXiv**](https://arxiv.org/abs/2310.10835) | [**webpage**](http://imaging.cms.caltech.edu/pmc/)). 

## Environment
Setup the environment.
```
conda env create --file pmc.yml
```
Activate the environment.
```
conda activate pmc
```
Deactivate the environment.
```
conda deactivate
```

## Checkpoints
The pre-trained checkpoints are available [**here**](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/ysun214_jh_edu/Ek2vadSp7u1Ct2qaBQX-b10B9LCBo_Xjxq98pF-kifBSSA?e=WHMMWm). Note that the checkpoints need to be placed in the proper fold for loading.

Create the folder
```
mkdir score_ckpt
```
Move the file to the folder
```
mv dir/to/ckpt ./score_ckpt
```

## Run
To run the code, simply type
```
python run_pmc.py -c dir/to/config.yml
```
Please add or configurate the data loader (`pmc/test_datasets/`) to allow proper loading of your own data.

## Citation
```
@ARTICLE{10645293,
  author={Sun, Yu and Wu, Zihui and Chen, Yifan and Feng, Berthy T. and Bouman, Katherine L.},
  journal={IEEE Transactions on Computational Imaging}, 
  title={Provable Probabilistic Imaging Using Score-Based Generative Priors}, 
  year={2024},
  volume={10},
  number={},
  pages={1290-1305},
  doi={10.1109/TCI.2024.3449114}}
```