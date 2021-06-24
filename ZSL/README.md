# GRaWD_ICCV21_Review
- Imaginative Walks: Generative Random Walk Deviation Loss for Improved Unseen Learning Representation Example Code - For ICCV21 Review Only.

- We provide the example code to reproduce the results for GAZSL+GRaWD on all textual-based datasets (i.e., CUB-wiki/NAB-wiki hard/easy splits.)

# Requirements
- Python 3.6
- Pytorch 1.6, torchvision 0.4.0
- sklearn, scipy, matplotlib, numpy, random, copy and other normal packages

# How to run?
## Preparing data
Download the [dataset CUBird and NABird](https://www.dropbox.com/s/9qovr86kgogkl6r/CUB_NAB_Data.zip).

Please put the uncompressed data to the folder "data"

## Training and Testing
We provide integrated code for training and testing. 

```
cd 'your main folder'

python train.py --dataset CUB --splitmode easy --exp_name 'CUB_easy_Rep' --rw_config_path ./configs/CUB_easy_Best_HPs.yml
python train.py --dataset CUB --splitmode hard --exp_name 'CUB_hard_Rep' --rw_config_path ./configs/CUB_hard_Best_HPs.yml
python train.py --dataset NAB --splitmode easy --exp_name 'NAB_easy_Rep' --rw_config_path ./configs/NAB_easy_Best_HPs.yml
python train.py --dataset NAB --splitmode hard --exp_name 'NAB_hard_Rep' --rw_config_path ./configs/NAB_hard_Best_HPs.yml
```

Here we provide the potential best hyper-parameters that can reproduce our reported results. You can refer to the logs under main folder for the training details. For each trial, the final performance may vary a little bit. Following standard setting, we report the best performance after k-trials. 