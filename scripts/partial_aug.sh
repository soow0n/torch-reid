aug=5
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=10
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=15
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=20
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=25
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=30
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=35
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1

aug=40
CUDA_VISIBLE_DEVICES=0 python scripts/main.py \
    --config-file configs/skt/aug_partial_pid.yaml \
    --project_name partial-aug --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/aug_partial_pid/aug${aug} \
    test.eval_freq 1 \
    test.eval_trainset True \
    test.save_pid_freq -1