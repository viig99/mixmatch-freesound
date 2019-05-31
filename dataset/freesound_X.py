import numpy as np
import torch
import librosa
from specAugment import spec_augment_pytorch
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from torchvision.transforms import *
from transforms import *

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def label_binarizer(data):
    lb = preprocessing.MultiLabelBinarizer()
    lb.fit(data)
    return lb

def spec_augment(audio_spec):
    warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=audio_spec)
    return warped_masked_spectrogram

def collate_fn(batch):
    specs, labels = zip(*batch)
    padded_specs = []
    for spec in specs:
        padded_spec = spec[..., ::4]
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
        padded_spec = spec[..., ::4]
        padded_specs1.append(padded_spec)
    padded_specs2 = []
    for spec in spec2:
        padded_spec = spec[..., ::4]
        padded_specs2.append(padded_spec)
    padded_specs1 = np.stack(padded_specs1, axis=0)[:,np.newaxis, :]
    padded_specs2 = np.stack(padded_specs2, axis=0)[:,np.newaxis, :]
    return (torch.from_numpy(padded_specs1), torch.from_numpy(padded_specs2)), None

def get_freesound():
    root = "/Users/vigi99/kaggle/freesound/data"
    # root = "/tts_data/kaggle/freesound/data"
    labelled_dir = os.path.join(root, "train_curated")
    unlabelled_dir = os.path.join(root, "train_noisy")
    test_dir = os.path.join(root, "test")

    labelled_df = pd.read_csv(os.path.join(root, "train_curated.csv"))
    unlabelled_df = pd.read_csv(os.path.join(root, "train_noisy.csv"))
    unlabelled_df = pd.read_csv(os.path.join(root, "train_noisy.csv"))

    labelled_files = [os.path.join(labelled_dir, fname) for fname in labelled_df.fname.values]
    labelled_labels = [label.split(",") for label in labelled_df.labels.values]
    unlabelled_files = [os.path.join(unlabelled_dir, fname) for fname in unlabelled_df.fname.values]
    unlabelled_labels = [label.split(",") for label in unlabelled_df.labels.values]

    lb = label_binarizer(labelled_df.labels.values.tolist() + unlabelled_df.labels.values.tolist())

    labelled_files_train, labelled_files_val, labelled_labels_train, labelled_labels_val = train_test_split(labelled_files, labelled_labels, test_size=0.1)

    data_aug_transform = Compose([ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(30), ToSTFT(), StretchAudioOnSTFT(), TimeshiftAudioOnSTFT(), FixSTFTDimension()])
    train_feature_transform = Compose([ToMelSpectrogramFromSTFT(n_mels=80), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
    valid_feature_transform = Compose([ToMelSpectrogram(n_mels=80), ToTensor('mel_spectrogram', 'input')])

    train_transforms = Compose([LoadAudio(), data_aug_transform, train_feature_transform]))
    valid_transforms = Compose([LoadAudio(), FixAudioLength(30), valid_feature_transform]))

    train_labeled_dataset = Freesound_labelled(labelled_files_train, labelled_labels_train, lb, transform=train_transforms)
    train_unlabeled_dataset = Freesound_unlabelled(unlabelled_files, unlabelled_labels, lb, transform=TransformTwice(data_aug_transform))
    val_dataset = Freesound_labelled(labelled_files_val, labelled_labels_val, lb, transform=valid_transforms)
    test_dataset = val_dataset

    print (f"#Labeled: {len(labelled_files_train)} #Unlabeled: {len(unlabelled_files)} #Val: {len(labelled_files_val)} #Num Classes: {len(lb.classes_)}")
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
        self.labels = None