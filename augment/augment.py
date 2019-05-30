import librosa                                                                                                                                                              
from specAugment import spec_augment_pytorch

def get_warped_masked_spectrogram(audio_file_path):
  audio, sampling_rate = librosa.load(audio_file_path)
  mel_spectrogram = librosa.feature.melspectrogram(y=audio,sr=sampling_rate,n_mels=256,hop_length=128,fmax=8000)
  warped_masked_spectrogram = spec_augment_pytorch.spec_augment(mel_spectrogram=mel_spectrogram)
  return warped_masked_spectrogram


