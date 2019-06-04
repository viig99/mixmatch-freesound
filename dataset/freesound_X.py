import numpy as np
import torch
import librosa
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchvision.transforms import *
from transforms import *

def label_binarizer(data):
    lb = preprocessing.MultiLabelBinarizer()
    lb.fit(data)
    return lb

def collate_fn(batch):
    specs, labels = zip(*batch)
    padded_specs = []
    for spec in specs:
        padded_spec = spec[..., ::2]
        padded_specs.append(padded_spec)
    padded_specs = np.stack(padded_specs, axis=0)[:,np.newaxis, :]
    labels = np.stack(labels, axis=0)
    return torch.from_numpy(padded_specs), torch.from_numpy(labels)

def collate_fn_unlabbelled(batch):
    spec, labels = zip(*batch)
    spec1 = [x[0] for x in spec]
    spec2 = [x[1] for x in spec]
    padded_specs1 = []
    for spec in spec1:
        padded_spec = spec[..., ::2]
        padded_specs1.append(padded_spec)
    padded_specs2 = []
    for spec in spec2:
        padded_spec = spec[..., ::2]
        padded_specs2.append(padded_spec)
    padded_specs1 = np.stack(padded_specs1, axis=0)[:,np.newaxis, :]
    padded_specs2 = np.stack(padded_specs2, axis=0)[:,np.newaxis, :]
    labels = np.stack(labels, axis=0)
    return (torch.from_numpy(padded_specs1), torch.from_numpy(padded_specs2)), torch.from_numpy(labels)

def get_freesound():
    # root = "/Users/vigi99/kaggle/freesound/data"
    root = "/tts_data/kaggle/freesound/data"
    labelled_dir = os.path.join(root, "train_curated")
    unlabelled_dir = os.path.join(root, "train_noisy")
    test_dir = os.path.join(root, "test")

    labelled_train_df = pd.read_csv(os.path.join(root, "train_curated.csv_train"))
    labelled_val_df = pd.read_csv(os.path.join(root, "train_curated.csv_test"))
    labelled_dev_df = pd.read_csv(os.path.join(root, "train_curated.csv_dev"))
    unlabelled_df = pd.read_csv(os.path.join(root, "train_noisy.csv"))

    labelled_files_train = [os.path.join(labelled_dir, fname) for fname in labelled_train_df.fname.values]
    labelled_labels_train = [label.split(",") for label in labelled_train_df.labels.values]

    labelled_files_val = [os.path.join(labelled_dir, fname) for fname in labelled_val_df.fname.values]
    labelled_labels_val = [label.split(",") for label in labelled_val_df.labels.values]

    labelled_files_dev = [os.path.join(labelled_dir, fname) for fname in labelled_dev_df.fname.values]
    labelled_labels_dev = [label.split(",") for label in labelled_dev_df.labels.values]

    unlabelled_files = [os.path.join(unlabelled_dir, fname) for fname in unlabelled_df.fname.values]
    unlabelled_labels = [label.split(",") for label in unlabelled_df.labels.values]

    lb = label_binarizer(labelled_labels_train + labelled_labels_val + labelled_labels_dev + unlabelled_labels)

    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(30), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=80), DeleteSTFT(), SpecAugmentOnMel(), ToTensor('mel_spectrogram')])
    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=80), ToTensor('mel_spectrogram')])

    train_transforms = Compose([LoadAudio(), data_aug_transform, train_feature_transform])
    valid_transforms = Compose([LoadAudio(), FixAudioLength(30), valid_feature_transform])

    train_labeled_dataset = Freesound_labelled(labelled_files_train, labelled_labels_train, lb, transform=train_transforms)
    train_unlabeled_dataset = Freesound_unlabelled(unlabelled_files, unlabelled_labels, lb, transform=TransformTwice(train_transforms))
    val_dataset = Freesound_labelled(labelled_files_val, labelled_labels_val, lb, transform=valid_transforms)
    test_dataset = Freesound_labelled(labelled_files_dev, labelled_labels_dev, lb, transform=valid_transforms)

    print (f"#Labeled: {len(labelled_files_train)} #Unlabeled: {len(unlabelled_files)} #Val: {len(labelled_files_val)} #Dev: {len(labelled_files_dev)} #Num Classes: {len(lb.classes_)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, len(lb.classes_)

class Freesound_labelled(Dataset):
    def __init__(self, files, labels, lb, transform=None):
        self.files = files
        self.labels = labels
        self.transform = transform
        if self.labels is not None:
            self.labels = np.array(lb.transform(self.labels), dtype='float32')

    def __getitem__(self, index):
        audio = self.files[index]
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = None
        if self.transform is not None:
            spec = self.transform(audio)
        return spec, label

    def __len__(self):
        return len(self.files)

class Freesound_unlabelled(Freesound_labelled):
    def __init__(self, files, labels, lb, transform=None):
        super(Freesound_unlabelled, self).__init__(files, labels, lb, transform=transform)