train_dir=/home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/models/baseline_v2_noisy

rm -rf $train_dir/* 
export CUDA_VISIBLE_DEVICES=1
python runner.py \
    --mode train \
    --model mobilenet-v1 \
    --class_map_path /home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/class_map.csv \
    --train_clip_dir /home/gpu_user/newdisk2/hari/data/freesound/train_noisy \
    --train_csv_path /home/gpu_user/newdisk2/hari/data/freesound/train_noisy.csv \
    --train_dir $train_dir \
    --hparams batch_size=64,lr=1e-4,dropout=0.8,lsmooth=0.3,global_pool='max' \
