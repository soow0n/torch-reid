# CUDA_VISIBLE_DEVICES=3 python scripts/main.py --config-file configs/skt/aug_all_pid.yaml \
#     data.aug_per_pid 25 \
#     data.save_dir '/mnt/data4/soowon/skt-intern/aug_all_pid/aug25' \
#     test.eval_freq 5 \
#     test.eval_trainset True \
#     test.save_pid_freq 10

aug=25
pretrained=luperson
CUDA_VISIBLE_DEVICES=3 python scripts/main.py \
    --config-file configs/skt/${pretrained}_finetune.yaml \
    --project_name $pretrained --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/aug${aug}
    
pretrained=sysu30k
CUDA_VISIBLE_DEVICES=3 python scripts/main.py \
    --config-file configs/skt/${pretrained}_finetune.yaml \
    --project_name $pretrained --exp_name aug${aug} \
    data.aug_per_pid $aug \
    data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/aug${aug}