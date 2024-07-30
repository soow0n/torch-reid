#!/bin/bash

project_name='Duke-Aug'
configs='configs/skt/duke_baseline.yaml'

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_baseline \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 0 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/baseline \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug5 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 5 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug5 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug10 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 10 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug10 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug15 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 15 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug15 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug20 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 20 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug20 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug25 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 25 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug25 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug30 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 30 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug30 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug35 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 35 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug35 \

CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --exp_name duke_aug40 \
    --project_name $project_name \
    --config-file $configs \
    data.aug_per_pid 40 \
    data.save_dir /mnt/data4/woojeong_skt/torchreid/dukemtmc/lr2/aug40 \