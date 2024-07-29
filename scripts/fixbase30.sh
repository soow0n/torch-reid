fixbase=30

# CUDA_VISIBLE_DEVICES=3 python scripts/main.py \
#     --config-file configs/skt/aug_all_pid.yaml \
#     --project_name augment-all-pid --exp_name baseline-fixbase${fixbase} \
#     data.aug_per_pid 0 \
#     data.save_dir /mnt/data4/soowon/skt-intern/aug_all_pid/baseline-fixbase${fixbase} \
#     test.eval_freq 1 \
#     test.eval_trainset True \
#     train.fixbase_epoch $fixbase


# pretrained=luperson
# CUDA_VISIBLE_DEVICES=3 python scripts/main.py \
#     --config-file configs/skt/${pretrained}_finetune.yaml \
#     --project_name $pretrained --exp_name baseline-fixbase${fixbase} \
#     data.aug_per_pid 0 \
#     data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline-fixbase${fixbase} \
#     test.eval_freq 1 \
#     test.eval_trainset True \
#     train.fixbase_epoch $fixbase
    

pretrained=sysu30k
CUDA_VISIBLE_DEVICES=3 python scripts/main.py \
    --config-file configs/skt/${pretrained}_finetune.yaml \
    --project_name $pretrained --exp_name baseline-fixbase${fixbase} \
    data.aug_per_pid 0 \
    data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline-fixbase${fixbase} \
    test.eval_freq 1 \
    test.eval_trainset True \
    train.fixbase_epoch $fixbase
