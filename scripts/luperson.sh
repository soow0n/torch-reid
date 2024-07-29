pretrained=luperson
CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
    --config-file configs/skt/${pretrained}_finetune.yaml \
    --project_name $pretrained --exp_name baseline-fixbase0 \
    data.aug_per_pid 0 \
    data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline \
    train.fixbase_epoch 0
    
# pretrained=sysu30k
# CUDA_VISIBLE_DEVICES=2 python scripts/main.py \
#     --config-file configs/skt/${pretrained}_finetune.yaml \
#     --project_name $pretrained --exp_name baseline-fixbase0 \
#     data.aug_per_pid 0 \
#     data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline-fixbase0 \
#     train.fixbase_epoch 0