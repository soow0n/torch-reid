aug=0
CUDA_VISIBLE_DEVICES=2 python scripts/main.py \
    --config-file configs/skt/market_baseline.yaml \
    --project_name MARS --exp_name baseline \
    data.save_dir /mnt/data4/soowon/skt-intern/MARS/baseline \
    data.aug_per_pid $aug \
    data.sample_mars False
