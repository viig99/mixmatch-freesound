import os
import numpy as np
import librosa
from transforms import *
from tqdm import tqdm
import pandas as pd
from torchvision.transforms import *
from dask import compute, delayed
from functools import partial

def get_mel_for_file(file):
    data_aug_transform = Compose([LoadAudio(), FixAudioLength(30), ToMelSpectrogram(n_mels=80)])
    return data_aug_transform(file)

def process_file(dirpath, file, pbar):
    fname = os.path.basename(file).replace('wav', 'npy')
    fpath = os.path.join(dirpath, fname)
    if not os.path.exists(fpath):
        spec = get_mel_for_file(file)['mel_spectrogram']
        np.save(fpath, spec)
    else:
        spec = np.load(fpath)
    pbar.update()
    return spec.mean(), spec.std()

def dask_apply(process, dirpath, inputs):
    pbar = tqdm(total=len(inputs))
    process = partial(process, pbar=pbar)
    values = [delayed(process)(dirpath, x) for x in inputs]
    results = compute(*values, scheduler='threads')
    pbar.close()
    return results

def process_files(dirpath, files):
    results = dask_apply(process_file, dirpath, files)
    means, stds = zip(*results)
    return means, stds

if __name__ == "__main__":
    root = "/tts_data/kaggle/freesound/data"
    labelled_dir = os.path.join(root, "train_curated")
    unlabelled_dir = os.path.join(root, "train_noisy")
    test_dir = os.path.join(root, "test")

    labelled_spec_dir = os.path.join(root, "train_curated_spec")
    unlabelled_spec_dir = os.path.join(root, "train_noisy_spec")
    test_spec_dir = os.path.join(root, "test_spec")

    os.makedirs(labelled_spec_dir, exist_ok=True)
    os.makedirs(unlabelled_spec_dir, exist_ok=True)

    labelled_train_df = pd.read_csv(os.path.join(root, "train_curated.csv_train"))
    labelled_val_df = pd.read_csv(os.path.join(root, "train_curated.csv_test"))
    labelled_dev_df = pd.read_csv(os.path.join(root, "train_curated.csv_dev"))
    unlabelled_df = pd.read_csv(os.path.join(root, "train_noisy.csv"))

    labelled_files_train = [os.path.join(labelled_dir, fname) for fname in labelled_train_df.fname.values]
    labelled_files_val = [os.path.join(labelled_dir, fname) for fname in labelled_val_df.fname.values]
    labelled_files_dev = [os.path.join(labelled_dir, fname) for fname in labelled_dev_df.fname.values]
    unlabelled_files = [os.path.join(unlabelled_dir, fname) for fname in unlabelled_df.fname.values]

    means_train, std_train = process_files(labelled_spec_dir, labelled_files_train + labelled_files_val + labelled_files_dev)
    means_noisy, std_noisy = process_files(unlabelled_spec_dir, unlabelled_files)

    print(f"Mean: {np.mean(means_train + means_noisy)}, Std: {np.mean(std_train + std_noisy)}")