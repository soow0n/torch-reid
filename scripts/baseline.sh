CUDA_VISIBLE_DEVICES=1 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 0 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/baseline' \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq 5