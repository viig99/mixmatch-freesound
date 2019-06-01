import random
from random import sample
from shutil import copyfile

random.seed(9)

base_dir = '/home/gpu_user/newdisk2/hari/data/freesound/'
train_set = 'train_curated'
train_dir = base_dir + train_set + '/'
train_csv = base_dir + train_set + '.csv'

output_csv_train = train_csv + '_train'
output_csv_dev = train_csv + '_dev'
output_csv_test = train_csv + '_test'

output_base_dir = base_dir + 'split_train_curated/'
audio_train = output_base_dir + 'train/'
audio_dev = output_base_dir + 'dev/'
audio_test = output_base_dir + 'test/'

lines = open(train_csv).readlines()
lines = [line.strip() for line in lines]

header = lines[0]
lines = lines[1:]

bad_audio_files = ['f76181c4.wav', '77b925c2.wav', '6a1f682a.wav', 'c7db12aa.wav', '7752cc8a.wav', '1d44b0bd.wav']

assert lines[0].split(',')[0].endswith('.wav')
print(lines[0].split(',')[0].strip())

line_cnt_before_filter = len(lines)

lines = [line for line in lines if line.split(',')[0].strip() not in bad_audio_files]

line_cnt_after_filter = len(lines)

print(line_cnt_before_filter, len(bad_audio_files), line_cnt_after_filter)

assert line_cnt_before_filter - len(bad_audio_files) == line_cnt_after_filter

random.shuffle(lines)

train_prop, dev_prop, test_prop = 0.9, 0.05, 0.05

train_set_size = int(len(lines)*train_prop)
dev_set_size = int(len(lines)*dev_prop)

train_lines = lines[:train_set_size]
dev_lines = lines[train_set_size:(train_set_size + dev_set_size)]
test_lines = lines[(train_set_size + dev_set_size):]

def create_set(output_csv, lines_sample, output_audio_dir):
 audio_set = set()
 with open(output_csv, 'w') as ofp:
  ofp.write(header + '\n')
  for line in lines_sample:
    ofp.write(line + '\n')
    audio_file = line.split(',')[0]
    audio_set.add(audio_file)
    output_audio_path = output_audio_dir + audio_file
    src_audio_path = train_dir + audio_file
    copyfile(src_audio_path, output_audio_path)
 return audio_set

train_audio_set = create_set(output_csv_train, train_lines, audio_train)
dev_audio_set = create_set(output_csv_dev, dev_lines, audio_dev)
test_audio_set = create_set(output_csv_test, test_lines, audio_test)

assert len(train_audio_set.intersection(dev_audio_set)) == 0
assert len(train_audio_set.intersection(test_audio_set)) == 0
assert len(dev_audio_set.intersection(test_audio_set)) == 0

assert len(train_audio_set) + len(dev_audio_set) + len(test_audio_set) == len(lines)
