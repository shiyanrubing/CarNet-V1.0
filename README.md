# CarNet-V1.0
This project contains the source code of our paper ["Rethinking Lightweight Convolutional Neural Networks for Efficient and High-quality Pavement Crack Detection"](https://arxiv.org/abs/2109.05707).
In the paper, we propose CarNet which is a fast pavement crack detector which achieves excellent accuracy.
We also propose three new pavement crack datasets, namely [Sun520, Rain365, and BJN260](https://github.com/shiyanrubing/CarNet_databases), to facilitate related research in the community.

# Documnetation

## Install
Clone repo and install requirements.txt in a Python=3.6 environment, including PyTorch=1.6.

```
git clone https://github.com/shiyanrubing/CarNet-V1.0  # clone
cd CarNet-V1.0
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
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

# Citation

```
@article{li2021fast,
    title={Fast and Accurate Road Crack Detection Based on Adaptive Cost-Sensitive Loss Function},
    author={Kai Li and Bo Wang and Yingjie Tian and Zhiquan Qi},
    journal={IEEE Transactions on Cybernetics},
    pages={1-12},
    year={2021},
    doi={10.1109/TCYB.2021.3103885}
}

@article{li2023fast,
   title={{Rethinking Lightweight Convolutional Neural Networks for Efficient and High-quality Pavement Crack Detection}},
   author={Kai Li, Jie Yang, Siwei Ma, Bo Wang, Shanshe Wang, Yingjie Tian, and Zhiquan Qi},
   journal="arXiv preprint arXiv:2109.05707",
   year={2023}
  }
  ```
