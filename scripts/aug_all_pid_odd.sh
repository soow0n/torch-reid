CUDA_VISIBLE_DEVICES=1 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 5 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug5' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10

CUDA_VISIBLE_DEVICES=1 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 15 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug15' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10


CUDA_VISIBLE_DEVICES=1 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 25 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug25' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10


CUDA_VISIBLE_DEVICES=1 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
    data.aug_per_pid 35 \
    data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug35' \
    test.eval_freq 5 \
    test.eval_trainset True \
    test.save_pid_freq 10

