# CarNet-V1.0
This project contains the source code of our paper ["Rethinking Lightweight Convolutional Neural Networks for Efficient and High-quality Pavement Crack Detection"](https://arxiv.org/abs/2109.05707).
In the paper, we propose CarNet which is a fast pavement crack detector which achieves excellent accuracy.

# Documnetation

## Install
Clone repo and install requirements.txt in a Python=3.6 environment, including PyTorch=1.6.

```
git clone https://github.com/shiyanrubing/CarNet-V1.0  # clone
cd CarNet-V1.0
pip install -r requirements.txt  # install
```

## Datasets
Download datasets from [here](https://github.com/shiyanrubing/CarNet_databases)

## Train
Configure the specific dataset in the *cfg.py*, and run *crack_train.py* in command line 

```
python crack_train.py
```

## Test
Run *test.py* in command line

```
python test.py
```
