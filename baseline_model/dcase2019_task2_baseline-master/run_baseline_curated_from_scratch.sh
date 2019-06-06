train_dir=/home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/models/baseline_v2_curated_from_scratch

rm -rf $train_dir/* 
export CUDA_VISIBLE_DEVICES=2
python runner.py \
    --mode train \
    --model mobilenet-v1 \
    --class_map_path /home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/class_map.csv \
    --train_clip_dir /home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/train \
    --train_csv_path /home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/train_curated.csv_train \
    --train_dir $train_dir \
    --hparams batch_size=64,lr=3e-3,lrdecay=0.94,dropout=0.6,lsmooth=0.1,global_pool='max' \
    --epoch_num_batches 70 
