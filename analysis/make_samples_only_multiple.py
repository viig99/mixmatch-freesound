import random
from random import sample
from shutil import copyfile

random.seed(9)

base_dir = '/home/gpu_user/newdisk2/hari/data/freesound/'
train_set = 'train_noisy'
train_dir = base_dir + train_set + '/'
train_csv = base_dir + 'processed/' + train_set + '.csv'

output_csv = train_csv + '_sample_multiple'
output_audio_dir = base_dir + 'processed/' + train_set + '_sample_multiple/'

lines = open(train_csv).readlines()
lines = [line.strip() for line in lines]

header = lines[0]
lines = lines[1:]

lines = [line for line in lines if line.count(",") > 4]

lines_sample = sample(lines, 50)

with open(output_csv, 'w') as ofp:
  ofp.write(header + '\n')
  for line in lines_sample:
    ofp.write(line + '\n')
    audio_file = line.split(',')[3]
    output_audio_path = output_audio_dir + audio_file
    src_audio_path = train_dir + audio_file
    copyfile(src_audio_path, output_audio_path)


