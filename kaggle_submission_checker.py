import random
import numpy as np
import librosa
from torch.utils.data import Dataset
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import pickle
from torchvision.transforms import Compose
import torch.utils.data as data
from timeit import default_timer as timer
from tqdm import tqdm
from utils.eval import lwlrap_accumulator

def should_apply_transform(prob=0.5):
    """Transforms are only randomly applied with the given probability."""
    return random.random() < prob

class LoadAudio(object):
    """Loads an audio into a numpy array."""

    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def __call__(self, path):
        data = {'path': path}
        if path:
            samples, sample_rate = librosa.load(path, self.sample_rate)
        else:
            # silence
            sample_rate = self.sample_rate
            samples = np.zeros(sample_rate, dtype=np.float32)
        data['samples'] = samples
        data['sample_rate'] = sample_rate
        return data

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

class FixAudioLength(object):
    """Either pads or truncates an audio into a fixed length."""

    def __init__(self, time=1):
        self.time = time

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        length = int(self.time * sample_rate)
        if length < len(samples):
            data['samples'] = samples[:length]
        elif length > len(samples):
            times = math.floor(length / len(samples))
            samples = np.hstack([samples] * times)
            data['samples'] = np.pad(samples, (0, length - len(samples)), "constant")
        return data

class ChangeAmplitude(object):
    """Changes amplitude of an audio randomly."""

    def __init__(self, amplitude_range=(0.7, 1.1)):
        self.amplitude_range = amplitude_range

    def __call__(self, data):
        if not should_apply_transform():
            return data

        data['samples'] = data['samples'] * random.uniform(*self.amplitude_range)
        return data

class ChangeSpeedAndPitchAudio(object):
    """Change the speed of an audio. This transform also changes the pitch of the audio."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        scale = random.uniform(-self.max_scale, self.max_scale)
        speed_fac = 1.0  / (1 + scale)
        data['samples'] = np.interp(np.arange(0, len(samples), speed_fac), np.arange(0,len(samples)), samples).astype(np.float32)
        return data

class StretchAudio(object):
    """Stretches an audio randomly."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        scale = random.uniform(-self.max_scale, self.max_scale)
        data['samples'] = librosa.effects.time_stretch(data['samples'], 1+scale)
        return data

class TimeshiftAudio(object):
    """Shifts an audio randomly."""

    def __init__(self, max_shift_seconds=0.2):
        self.max_shift_seconds = max_shift_seconds

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        sample_rate = data['sample_rate']
        max_shift = (sample_rate * self.max_shift_seconds)
        shift = random.randint(-max_shift, max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        samples = np.pad(samples, (a, b), "constant")
        data['samples'] = samples[:len(samples) - a] if a else samples[b:]
        return data

class AddBackgroundNoise(Dataset):
    """Adds a random background noise."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        samples = data['samples']
        noise = random.choice(self.bg_dataset)['samples']
        percentage = random.uniform(0, self.max_percentage)
        data['samples'] = samples * (1 - percentage) + noise * percentage
        return data

class ToMelSpectrogram(object):
    """Creates the mel spectrogram from an audio. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        s = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=self.n_mels)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class ToTensor(object):
    """Converts into a tensor."""

    def __init__(self, np_name, normalize=None):
        self.np_name = np_name
        self.normalize = normalize

    def __call__(self, data):
        tensor = data[self.np_name].astype(np.float32)
        if self.normalize is not None:
            mean, std = self.normalize
            tensor -= mean
            tensor /= std
        return tensor

class ToSTFT(object):
    """Applies on an audio the short time fourier transform."""

    def __init__(self, n_fft=2048, hop_length=512):
        self.n_fft = n_fft
        self.hop_length = hop_length

    def __call__(self, data):
        samples = data['samples']
        sample_rate = data['sample_rate']
        data['n_fft'] = self.n_fft
        data['hop_length'] = self.hop_length
        data['stft'] = librosa.stft(samples, n_fft=self.n_fft, hop_length=self.hop_length)
        data['stft_shape'] = data['stft'].shape
        return data

class StretchAudioOnSTFT(object):
    """Stretches an audio on the frequency domain."""

    def __init__(self, max_scale=0.2):
        self.max_scale = max_scale

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        sample_rate = data['sample_rate']
        hop_length = data['hop_length']
        scale = random.uniform(-self.max_scale, self.max_scale)
        stft_stretch = librosa.core.phase_vocoder(stft, 1+scale, hop_length=hop_length)
        data['stft'] = stft_stretch
        return data

class TimeshiftAudioOnSTFT(object):
    """A simple timeshift on the frequency domain without multiplying with exp."""

    def __init__(self, max_shift=8):
        self.max_shift = max_shift

    def __call__(self, data):
        if not should_apply_transform():
            return data

        stft = data['stft']
        shift = random.randint(-self.max_shift, self.max_shift)
        a = -min(0, shift)
        b = max(0, shift)
        stft = np.pad(stft, ((0, 0), (a, b)), "constant")
        if a == 0:
            stft = stft[:,b:]
        else:
            stft = stft[:,0:-a]
        data['stft'] = stft
        return data

class AddBackgroundNoiseOnSTFT(Dataset):
    """Adds a random background noise on the frequency domain."""

    def __init__(self, bg_dataset, max_percentage=0.45):
        self.bg_dataset = bg_dataset
        self.max_percentage = max_percentage

    def __call__(self, data):
        if not should_apply_transform():
            return data

        noise = random.choice(self.bg_dataset)['stft']
        percentage = random.uniform(0, self.max_percentage)
        data['stft'] = data['stft'] * (1 - percentage) + noise * percentage
        return data

class SpecAugmentOnMel(object):
    def __call__(self, data):
        if not should_apply_transform():
            return data


        data['mel_spectrogram'] = spec_augment_pytorch.spec_augment(mel_spectrogram=data['mel_spectrogram'])
        return data

class FixSTFTDimension(object):
    """Either pads or truncates in the time axis on the frequency domain, applied after stretching, time shifting etc."""

    def __call__(self, data):
        stft = data['stft']
        t_len = stft.shape[1]
        orig_t_len = data['stft_shape'][1]
        if t_len > orig_t_len:
            stft = stft[:,0:orig_t_len]
        elif t_len < orig_t_len:
            stft = np.pad(stft, ((0, 0), (0, orig_t_len-t_len)), "constant")

        data['stft'] = stft
        return data

class ToMelSpectrogramFromSTFT(object):
    """Creates the mel spectrogram from the short time fourier transform of a file. The result is a 32x32 matrix."""

    def __init__(self, n_mels=32):
        self.n_mels = n_mels

    def __call__(self, data):
        stft = data['stft']
        sample_rate = data['sample_rate']
        n_fft = data['n_fft']
        mel_basis = librosa.filters.mel(sample_rate, n_fft, self.n_mels)
        s = np.dot(mel_basis, np.abs(stft)**2.0)
        data['mel_spectrogram'] = librosa.power_to_db(s, ref=np.max)
        return data

class DeleteSTFT(object):
    """Pytorch doesn't like complex numbers, use this transform to remove STFT after computing the mel spectrogram."""

    def __call__(self, data):
        del data['stft']
        return data

class AudioFromSTFT(object):
    """Inverse short time fourier transform."""

    def __call__(self, data):
        stft = data['stft']
        data['istft_samples'] = librosa.core.istft(stft, dtype=data['samples'].dtype)
        return data

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

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=4, dropRate=0.1):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(1, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

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

def collate_fn(batch):
    specs, labels = zip(*batch)
    padded_specs = []
    for spec in specs:
        padded_spec = spec[..., ::2]
        padded_specs.append(padded_spec)
    padded_specs = np.stack(padded_specs, axis=0)[:,np.newaxis, :]
    labels = np.stack(labels, axis=0)
    return torch.from_numpy(padded_specs), torch.from_numpy(labels)

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
model_vals = torch.load('result/model_best.pth.tar', map_location='cpu')['ema_state_dict']
pickle.dump(model_vals, open('result/weights.pk', 'wb'))
'''

# GPU Server
test_path = os.path.abspath('/tts_data/split_train_curated/split_train_curated/test')
model_path = os.path.abspath('result/weights_v2.pk')
sample_submission_file = 'submission/submission_split_train_test.csv'
lb_path = os.path.abspath('submission/lb.pk')
correct_answers = os.path.abspath('/tts_data/split_train_curated/split_train_curated/train_curated.csv_test')
df = pd.read_csv(correct_answers)

batch_size = 8

lb = pickle.load(open(lb_path, 'rb'))
correct_labels = [labels.split(',') for labels in df['labels'].values]

# file_paths = [os.path.join(test_path, file) for file in os.listdir(test_path)]
file_paths = [os.path.join(test_path, file) for file in df['fname'].values]
valid_feature_transform = Compose([ToMelSpectrogram(n_mels=80), ToTensor('mel_spectrogram')])
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