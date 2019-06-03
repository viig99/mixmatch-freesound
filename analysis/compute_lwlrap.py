import numpy as np
import os

submission_csv = '/home/gpu_user/newdisk2/hari/data/freesound/airtel_results/submission_split_train_dev.csv'

audio_dir = '/home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/dev/'

ground_truth_csv = '/home/gpu_user/newdisk2/hari/data/freesound/split_train_curated/train_curated.csv_dev'

submission_lines = open(submission_csv).readlines()
submission_lines = [l.strip() for l in submission_lines]

submission_header = submission_lines[0]
assert submission_header.split(',')[0] == "fname"

submission_lines = submission_lines[1:]

audio_details = {}

for line in submission_lines:
 audio_file = line.split(',')[0]
 assert audio_file.endswith('.wav')
 probs_str = line[(len(audio_file) + 1):]
 probs_str_list = probs_str.split(',')
 #print('len:', len(probs_str_list))
 #print(probs_str_list)
 assert len(probs_str_list) == 80
 probs = [float(p) for p in probs_str_list]
 assert len(probs) == 80
 probs_np = np.array(probs).reshape(1,80)
 if audio_file not in audio_details:
   audio_details[audio_file] = {}
 audio_details[audio_file]['pred_array'] = probs_np

label_str_to_index = {}
label_str_list = submission_header.split(',')[1:]
assert len(label_str_list) == 80
for i, label_str in enumerate(label_str_list):
  label_str_to_index[label_str] = i 

gt_lines = open(ground_truth_csv).readlines()
gt_lines = [l.strip() for l in gt_lines]

gt_header = gt_lines[0]
assert gt_header.split(',')[0] == "fname"

gt_lines = gt_lines[1:] 

assert len(gt_lines) == len(submission_lines)

for line in gt_lines:
  audio_file = line.split(',')[0]
  assert audio_file.endswith('.wav')
  ground_truth_labels = line[(len(audio_file) + 1):]
  ground_truth_labels = ground_truth_labels.strip('"')
  ground_truth_label_list = ground_truth_labels.split(',')
  ground_truth_np = np.zeros((1,80))
  for ground_truth_label in ground_truth_label_list:
    ground_truth_np[0, label_str_to_index[ground_truth_label]] = 1
  audio_details[audio_file]['gt_array'] = ground_truth_np

audio_files = os.listdir(audio_dir)
assert len(audio_files) == len(submission_lines)

pred_np_list = []
gt_np_list = []

for audio_file in audio_files:
  pred_np_list.append(audio_details[audio_file]['pred_array'])
  gt_np_list.append(audio_details[audio_file]['gt_array'])

pred_np_stacked = np.vstack(tuple(pred_np_list))
gt_np_stacked = np.vstack(tuple(gt_np_list))  

######

import sklearn.metrics
    
# Core calculation of label precisions for one test sample.

def _one_sample_positive_class_precisions(scores, truth):
  """Calculate precisions for each true class for a single sample.
  
  Args:
    scores: np.array of (num_classes,) giving the individual classifier scores.
    truth: np.array of (num_classes,) bools indicating which classes are true.

  Returns:
    pos_class_indices: np.array of indices of the true classes for this sample.
    pos_class_precisions: np.array of precisions corresponding to each of those
      classes.
  """
  num_classes = scores.shape[0]
  pos_class_indices = np.flatnonzero(truth > 0)
  # Only calculate precisions if there are some true classes.
  if not len(pos_class_indices):
    return pos_class_indices, np.zeros(0)
  # Retrieval list of classes for this sample. 
  retrieved_classes = np.argsort(scores)[::-1]
  # class_rankings[top_scoring_class_index] == 0 etc.
  class_rankings = np.zeros(num_classes, dtype=np.int)
  class_rankings[retrieved_classes] = range(num_classes)
  # Which of these is a true label?
  retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
  retrieved_class_true[class_rankings[pos_class_indices]] = True
  # Num hits for every truncated retrieval list.
  retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
  # Precision of retrieval list truncated at each hit, in order of pos_labels.
  precision_at_hits = (
      retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 
      (1 + class_rankings[pos_class_indices].astype(np.float)))
  return pos_class_indices, precision_at_hits

# All-in-one calculation of per-class lwlrap.

def calculate_per_class_lwlrap(truth, scores):
  """Calculate label-weighted label-ranking average precision.
  
  Arguments:
    truth: np.array of (num_samples, num_classes) giving boolean ground-truth
      of presence of that class in that sample.
    scores: np.array of (num_samples, num_classes) giving the classifier-under-
      test's real-valued score for each class for each sample.
  
  Returns:
    per_class_lwlrap: np.array of (num_classes,) giving the lwlrap for each 
      class.
    weight_per_class: np.array of (num_classes,) giving the prior of each 
      class within the truth labels.  Then the overall unbalanced lwlrap is 
      simply np.sum(per_class_lwlrap * weight_per_class)
  """
  assert truth.shape == scores.shape
  num_samples, num_classes = scores.shape
  # Space to store a distinct precision value for each class on each sample.
  # Only the classes that are true for each sample will be filled in.
  precisions_for_samples_by_classes = np.zeros((num_samples, num_classes))
  for sample_num in range(num_samples):
    pos_class_indices, precision_at_hits = (
      _one_sample_positive_class_precisions(scores[sample_num, :], 
                                            truth[sample_num, :]))
    precisions_for_samples_by_classes[sample_num, pos_class_indices] = (
        precision_at_hits)
  labels_per_class = np.sum(truth > 0, axis=0)
  weight_per_class = labels_per_class / float(np.sum(labels_per_class))
  # Form average of each column, i.e. all the precisions assigned to labels in
  # a particular class.
  per_class_lwlrap = (np.sum(precisions_for_samples_by_classes, axis=0) / 
                      np.maximum(1, labels_per_class))
  # overall_lwlrap = simple average of all the actual per-class, per-sample precisions
  #                = np.sum(precisions_for_samples_by_classes) / np.sum(precisions_for_samples_by_classes > 0)
  #           also = weighted mean of per-class lwlraps, weighted by class label prior across samples
  #                = np.sum(per_class_lwlrap * weight_per_class)
  return per_class_lwlrap, weight_per_class

# Calculate the overall lwlrap using sklearn.metrics function.

def calculate_overall_lwlrap_sklearn(truth, scores):
  """Calculate the overall lwlrap using sklearn.metrics.lrap."""
  # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
  sample_weight = np.sum(truth > 0, axis=1)
  nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
  overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
      truth[nonzero_weight_sample_indices, :] > 0, 
      scores[nonzero_weight_sample_indices, :], 
      sample_weight=sample_weight[nonzero_weight_sample_indices])
  return overall_lwlrap

# Accumulator object version.

class lwlrap_accumulator(object):
  """Accumulate batches of test samples into per-class and overall lwlrap."""  

  def __init__(self):
    self.num_classes = 0
    self.total_num_samples = 0
  
  def accumulate_samples(self, batch_truth, batch_scores):
    """Cumulate a new batch of samples into the metric.
    
    Args:
      truth: np.array of (num_samples, num_classes) giving boolean
        ground-truth of presence of that class in that sample for this batch.
      scores: np.array of (num_samples, num_classes) giving the 
        classifier-under-test's real-valued score for each class for each
        sample.
    """
    assert batch_scores.shape == batch_truth.shape
    num_samples, num_classes = batch_truth.shape
    if not self.num_classes:
      self.num_classes = num_classes
      self._per_class_cumulative_precision = np.zeros(self.num_classes)
      self._per_class_cumulative_count = np.zeros(self.num_classes, 
                                                  dtype=np.int)
    assert num_classes == self.num_classes
    for truth, scores in zip(batch_truth, batch_scores):
      pos_class_indices, precision_at_hits = (
        _one_sample_positive_class_precisions(scores, truth))
      self._per_class_cumulative_precision[pos_class_indices] += (
        precision_at_hits)
      self._per_class_cumulative_count[pos_class_indices] += 1
    self.total_num_samples += num_samples

  def per_class_lwlrap(self):
    """Return a vector of the per-class lwlraps for the accumulated samples."""
    return (self._per_class_cumulative_precision / 
            np.maximum(1, self._per_class_cumulative_count))

  def per_class_weight(self):
    """Return a normalized weight vector for the contributions of each class."""
    return (self._per_class_cumulative_count / 
            float(np.sum(self._per_class_cumulative_count)))

  def overall_lwlrap(self):
    """Return the scalar overall lwlrap for cumulated samples."""
    return np.sum(self.per_class_lwlrap() * self.per_class_weight())

accumulator = lwlrap_accumulator()
accumulator.accumulate_samples(gt_np_stacked, pred_np_stacked)
print("cumulative_lwlrap=", accumulator.overall_lwlrap())
print("total_num_samples=", accumulator.total_num_samples)
######
