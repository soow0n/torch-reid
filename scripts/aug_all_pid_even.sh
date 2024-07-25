CUDA_VISIBLE_DEVICES=2 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 10 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug10' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10

CUDA_VISIBLE_DEVICES=2 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 20 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug20' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10


CUDA_VISIBLE_DEVICES=2 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 30 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug30' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10


CUDA_VISIBLE_DEVICES=2 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 40 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug40' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10

