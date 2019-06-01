import numpy as np

submission_csv = '/home/gpu_user/newdisk2/hari/data/freesound/airtel_results/submission_split_train_test.csv'

audio_dir = '/home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/test/'

ground_truth_csv = '/home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/train_curated.csv_test'

submission_lines = open(submission_csv).readlines()
submission_lines = [l.strip() for l in submission_lines]

submission_header = submission_lines[0]
assert submission_header.split(',')[0] == "fname"

submission_lines = submission_lines[1:]

audio_details = {}

for line in submission_lines:
 audio_file = line.split(',')[0]
 assert audio_file.endswith('.wav')
 probs_str = line[len(audio_file):]
 probs = [floar(p) for p in probs_str]
 assert len(probs) == 70
 probs_np = np.array(probs).reshape(1,70)
 if audio_file not in audio_details:
   audio_details[audio_file] = {}
 audio_details[audio_file]['pred_array'] = probs_np

label_str_to_index = {}
label_str_list = 

gt_lines = open(ground_truth_csv).readlines()
gt_lines = [l.strip() for l in gt_lines]

gt_header = gt_lines[0]
assert gt_header.split(',')[0] == "fname"

gt_lines = gt_lines[1:] 

for line in gt_lines:
  audio_file = line.split(',')[0]
  assert audio_file.endswith('.wav')
  ground_truth_labels = line[len(audio_file):]
  ground_truth_labels = ground_truth_labels.strip('"')
    

