import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import pickle
from torchvision.transforms import Compose
import torch.utils.data as data
from timeit import default_timer as timer
from tqdm import tqdm
from utils.eval import lwlrap_accumulator
from transforms import *
from dataset.freesound_X import Freesound_labelled

def create_model(ema=False):
    model = WideResNet(num_classes=80)
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

def load_state_dict(model, state_dict):
    new_state_dict = model.state_dict()
    for key in state_dict.keys():
        new_state_dict[key.replace('module.', '')] = state_dict[key]
    return new_state_dict

# Kaggle
# test_path = os.path.abspath('../input/freesound-audio-tagging-2019/test')
# model_path = os.path.abspath('../input/freesound2/weights.pk')
# sample_submission_file = 'submission.csv'
# lb_path = os.path.abspath('../input/freesound2/lb.pk')

# My PC
# test_path = os.path.abspath('/Users/vigi99/kaggle/freesound/data/test')
# model_path = os.path.abspath('result/weights.pk')
# sample_submission_file = 'submission.csv'
# lb_path = os.path.abspath('submission/lb.pk')

'''
import torch
import pickle
model_vals = torch.load('result_checkpoints/model_best.pth.tar', map_location='cpu')['ema_state_dict']
pickle.dump(model_vals, open('result/weights_noisy_mixmatch.pk', 'wb'))
'''

# GPU Server
if __name__ == "__main__": 
    test_path = os.path.abspath('/tts_data/kaggle/freesound/data/train_curated')
    model_path = os.path.abspath('result/weights_noisy_mixmatch.pk')
    sample_submission_file = 'submission/submission_split_train_dev.csv'
    lb_path = os.path.abspath('submission/lb.pk')
    correct_answers_1 = os.path.abspath('/tts_data/kaggle/freesound/data/train_curated.csv_dev')
    correct_answers_2 = os.path.abspath('/tts_data/kaggle/freesound/data/train_curated.csv_test')
    df = pd.concat([pd.read_csv(correct_answers_1), pd.read_csv(correct_answers_2)])

    batch_size = 8

    lb = pickle.load(open(lb_path, 'rb'))
    correct_labels = [labels.split(',') for labels in df['labels'].values]

    # file_paths = [os.path.join(test_path, file) for file in os.listdir(test_path)]
    file_paths = [os.path.join(test_path, file) for file in df['fname'].values]
    valid_feature_transform = Compose([ToSTFT(), ToMelSpectrogramFromSTFT(n_mels=80), ToTensor('mel_spectrogram')])
    valid_transforms = Compose([LoadAudio(), FixAudioLength(30), valid_feature_transform])

    val_dataset = Freesound_labelled(file_paths, correct_labels, lb, transform=valid_transforms)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=max(1, os.cpu_count() - 2), collate_fn=collate_fn)

    print('Loaded dataset')
    model = create_model(ema=True)
    best_model_state_dict = pickle.load(open(model_path, 'rb'))
    model.load_state_dict(load_state_dict(model, best_model_state_dict))
    print('Loaded Model')
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    result = []
    result_header = ['fname', *lb.classes_]
    lwlrap_acc = lwlrap_accumulator()

    with torch.no_grad():
        start_time = timer()
        for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            outputs = model(inputs)
            lwlrap_acc.accumulate_samples(targets, outputs)
            probs = torch.sigmoid(outputs).cpu().numpy().tolist()
            filenames = [os.path.basename(file) for file in np.array(val_dataset.files)[batch_size * batch_idx:batch_size * (batch_idx+1)].tolist()]
            for fname, prob in zip(filenames, probs):
                result.append([fname, *prob])
            print('Num of examples done {:d}'.format(batch_idx * batch_size))
        time_taken = timer() - start_time
        print('Total time taken was {:.4f} seconds'.format(time_taken))
        print("Time taken per example on cpu was {:.4f} seconds".format(time_taken / len(file_paths)))
        print("LRAP for test set was {:.4f}".format(lwlrap_acc.overall_lwlrap()))
        df = pd.DataFrame(result, columns=result_header)
        df.to_csv(sample_submission_file, index=False)