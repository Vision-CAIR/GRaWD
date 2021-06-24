#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J Rep_Best_hps_1
#SBATCH -o Rep_Best_hps_1.%J.out
#SBATCH -e Rep_Best_hps_1.%J.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gtx1080ti|rtx2080ti|v100]

#run the application:
cd 'your main folder'

#python train_GBU.py --dataset APY --preprocessing --exp_name 'aPY_A_Rep_Best_hps_1' --rw_config_path ./configs/aPY_Best_HPs.yml
#python train_GBU.py --dataset AWA2 --preprocessing --exp_name 'AWA2_A_Rep_Best_hps_1' --rw_config_path ./configs/AWA2_Best_HPs.yml
#python train_GBU.py --dataset SUN --preprocessing --exp_name 'SUN_A_Rep_Best_hps_1' --rw_config_path ./configs/SUN_Best_HPs.yml

python train.py --dataset CUB --splitmode easy --exp_name 'CUB_easy_Rep_Best_hps_1' --rw_config_path ./configs/CUB_easy_Best_HPs.yml
python train.py --dataset CUB --splitmode hard --exp_name 'CUB_hard_Rep_Best_hps_1' --rw_config_path ./configs/CUB_hard_Best_HPs.yml
python train.py --dataset NAB --splitmode easy --exp_name 'NAB_easy_Rep_Best_hps_1' --rw_config_path ./configs/NAB_easy_Best_HPs.yml
python train.py --dataset NAB --splitmode hard --exp_name 'NAB_hard_Rep_Best_hps_1' --rw_config_path ./configs/NAB_hard_Best_HPs.yml