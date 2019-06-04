export CUDA_VISIBLE_DEVICES=1

eval_dir=/home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/models/eval/eval_tmp
rm -rf $eval_dir/*

python  runner.py \
    --mode eval \
    --model mobilenet-v1 \
    --class_map_path /home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/class_map.csv \
    --eval_clip_dir /home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/dev \
    --eval_csv_path /home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/train_curated.csv_dev \
    --train_dir /home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/models/baseline_v2_curated_from_scratch \
    --eval_dir $eval_dir

#tensorboard --logdir=/home/gpu_user/newdisk2/hari/code/MixMatch-pytorch/baseline_model/dcase2019_task2_baseline-master/models/eval/baseline_v2_noisy --port 6008 --host 10.5.249.16
