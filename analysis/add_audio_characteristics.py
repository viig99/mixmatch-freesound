import librosa
from tqdm import tqdm

base_dir = '/home/gpu_user/newdisk2/hari/data/freesound/'
ourput_dir = base_dir + 'processed/'

train_sets = ['train_curated', 'train_noisy']

for train_set in train_sets:
  train_csv = base_dir + train_set + '.csv'
  train_audio_dir = base_dir + train_set + '/'
  
  output_train_csv = ourput_dir + train_set + '.csv'

  with open(output_train_csv, 'w') as ofp:
    header = True
    ctr = 0
    for line in tqdm(open(train_csv)):
      ctr += 1
      if ctr > 10:
        #break
        pass
      line = line.strip()
      if header:
        ofp.write('duration,sampling_rate,waveform_length,' + line + '\n')
        header = False
        continue
      audio_file = line.split(',')[0]
      audio_file_path = train_audio_dir + audio_file
      wave_form, sampling_rate = librosa.load(audio_file_path) 
      duration = librosa.get_duration(y=wave_form, sr=sampling_rate)     
      output_line = str(duration) + ',' + str(sampling_rate) + ',' + str(len(wave_form)) + ',' + line + '\n'
      ofp.write(output_line)

