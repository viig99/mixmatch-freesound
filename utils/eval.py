from __future__ import print_function, absolute_import
from sklearn.metrics import label_ranking_average_precision_score
import numpy as np
import torch

__all__ = ['accuracy', 'lwlrap_accumulator']

def accuracy(scores, truth, topk=(1,)):
	scores = torch.sigmoid(scores).numpy()
	truth = truth.numpy()
	sample_weight = np.sum(truth > 0, axis=1)
	nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
	overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
		truth[nonzero_weight_sample_indices, :] > 0,
		scores[nonzero_weight_sample_indices, :],
		sample_weight=sample_weight[nonzero_weight_sample_indices])
	return overall_lwlrap

def _one_sample_positive_class_precisions(scores, truth):
	num_classes = scores.shape[0]
	pos_class_indices = np.flatnonzero(truth > 0)
	if not len(pos_class_indices):
		return pos_class_indices, np.zeros(0)
	retrieved_classes = np.argsort(scores)[::-1]
	class_rankings = np.zeros(num_classes, dtype=np.int)
	class_rankings[retrieved_classes] = range(num_classes)
	retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
	retrieved_class_true[class_rankings[pos_class_indices]] = True
	retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
	precision_at_hits = (
		retrieved_cumulative_hits[class_rankings[pos_class_indices]] /
		(1 + class_rankings[pos_class_indices].astype(np.float)))
	return pos_class_indices, precision_at_hits

class lwlrap_accumulator(object):
	def __init__(self):
		self.num_classes = 0
		self.total_num_samples = 0

	def accumulate_samples(self, batch_truth, batch_scores):
		batch_scores = torch.sigmoid(batch_scores).numpy()
		batch_truth = batch_truth.numpy()
		assert batch_scores.shape == batch_truth.shape
		num_samples, num_classes = batch_truth.shape
		if not self.num_classes:
			self.num_classes = num_classes
			self._per_class_cumulative_precision = np.zeros(self.num_classes)
			self._per_class_cumulative_count = np.zeros(self.num_classes, dtype=np.int)
		assert num_classes == self.num_classes
		for truth, scores in zip(batch_truth, batch_scores):
			pos_class_indices, precision_at_hits = (
			_one_sample_positive_class_precisions(scores, truth))
			self._per_class_cumulative_precision[pos_class_indices] += (precision_at_hits)
			self._per_class_cumulative_count[pos_class_indices] += 1
		self.total_num_samples += num_samples

	def per_class_lwlrap(self):
		return (self._per_class_cumulative_precision / np.maximum(1, self._per_class_cumulative_count))

	def per_class_weight(self):
		return (self._per_class_cumulative_count / float(np.sum(self._per_class_cumulative_count)))

	def overall_lwlrap(self):
		return np.sum(self.per_class_lwlrap() * self.per_class_weight())