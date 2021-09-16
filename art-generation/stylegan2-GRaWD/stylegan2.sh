python -m torch.distributed.launch --nproc_per_node=1 --master_port=12895 train.py\
        --batch=32\
        --checkpoint_folder=styleGAN2\
        --n_sample=25\
        --size=256\
        --name_suffix=RW-W10-T10\
        --use_RW\
        --normalize_protos_scale=3.0\
        --RW_weight=10.0\
        --RW_tau=10\
        --no_pbar\
