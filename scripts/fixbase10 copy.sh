fixbase=10

# CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
#     --config-file configs/skt/aug_all_pid.yaml \
#     --project_name augment-all-pid --exp_name baseline-fixbase${fixbase} \
#     data.aug_per_pid 0 \
#     data.save_dir /mnt/data4/soowon/skt-intern/aug_all_pid/baseline-fixbase${fixbase} \
#     test.eval_freq 1 \
#     test.eval_trainset True \
#     train.fixbase_epoch $fixbase


# pretrained=luperson
# CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
#     --config-file configs/skt/${pretrained}_finetune.yaml \
#     --project_name $pretrained --exp_name baseline-fixbase${fixbase} \
#     data.aug_per_pid 0 \
#     data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline-fixbase${fixbase} \
#     test.eval_freq 1 \
#     test.eval_trainset True \
#     train.fixbase_epoch $fixbase
    

pretrained=sysu30k
CUDA_VISIBLE_DEVICES=1 python scripts/main.py \
    --config-file configs/skt/${pretrained}_finetune.yaml \
    --project_name $pretrained --exp_name baseline-fixbase${fixbase} \
    data.aug_per_pid 0 \
    data.save_dir /mnt/data4/soowon/skt-intern/${pretrained}_finetune/baseline-fixbase${fixbase} \
    test.eval_freq 1 \
    test.eval_trainset True \
    train.fixbase_epoch $fixbase
    [171, 360, 39, 17, 168, 440, 58, 1307, 846, 153, 258, 1150, 8, 40, 959, 78, 1013, 124, 1037, 1151, 1462, 14, 113, 1233, 300, 1222, 13, 908, 1143, 157, 675, 505, 454, 1036, 246, 1366, 938, 38, 156, 1044, 61, 542, 878, 170, 131, 154, 869, 1271, 74, 1073, 16, 618, 253, 401, 512, 262, 523, 66, 822, 1062, 230, 428, 316, 544, 1282, 1247, 103, 63, 574, 465, 532, 161, 1082, 1375, 502, 880, 1040, 1277, 732, 80, 458, 452, 1444, 632, 21, 686, 1130, 278, 888, 931, 94, 271, 716, 693, 713, 784, 595, 910, 130, 916, 192, 1005, 813, 144, 231, 319, 699, 240, 627, 9, 1446, 474, 1103, 1431, 425, 479, 824, 1144, 355, 965, 96, 145, 881, 725, 351, 182, 72, 791, 1224, 364, 1166, 745, 937, 800, 834, 815, 213, 1357, 92, 1460, 913, 60, 1401, 453, 974, 343, 198, 219, 75, 1085, 288, 275, 743, 165, 797, 866, 83, 934, 1482, 183, 260, 1190, 1333, 587, 1191, 322, 1192, 417, 25, 1499, 1481, 1328, 1214, 1199, 126, 1133, 200, 36, 471, 607, 777, 147, 62, 1118, 1161, 1361, 24, 1, 128, 342, 736, 801, 1450, 758, 543, 19, 1026, 1194, 302, 196, 838, 1022, 362, 1388, 89, 1340, 609, 1120, 119, 337, 152, 1478, 381, 439, 719, 29, 1061, 400, 1395, 768, 560, 146, 874, 483, 1180, 473, 244, 836, 395, 501, 218, 646, 747, 819, 187, 284, 1046, 510, 1181, 978, 263, 137, 1171, 226, 1164, 34, 1283, 582, 448, 538, 490, 1279, 1355, 1175, 1352, 786, 91, 531, 626, 373, 1183, 695, 55, 771, 624, 1065, 860, 217, 194, 727, 44, 1290, 315, 608, 1108, 50, 252, 1493, 559, 1301, 189, 396, 567, 33, 1035, 1477, 590, 109, 1014, 644, 756, 71, 740, 1156, 87, 1207, 365, 720, 1184, 205]
