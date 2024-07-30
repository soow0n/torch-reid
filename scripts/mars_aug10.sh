aug=10
CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
    --config-file configs/skt/market_baseline.yaml \
    --project_name MARS --exp_name mars-aug${aug} \
    data.save_dir /mnt/data4/soowon/skt-intern/MARS/mars-aug${aug} \
    data.aug_per_pid $aug \
    data.sample_mars True

CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
    --config-file configs/skt/market_baseline.yaml \
    --project_name MARS --exp_name diff-aug${aug} \
    data.save_dir /mnt/data4/soowon/skt-intern/MARS/diff-aug${aug} \
    data.aug_per_pid $aug \
    data.sample_mars False

