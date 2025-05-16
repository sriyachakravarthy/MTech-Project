"""
Implement dataloader for speech separation
Modifier: Shengkui Zhao
"""

import numpy as np
import math, os, csv
import torchaudio
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.distributed as dist
import soundfile as sf
from torch.utils.data import Dataset
import torch.utils.data as data
import os 
import sys
sys.path.append(os.path.dirname(__file__))

from dataloader.misc import read_and_config_file
import librosa
import random
EPS = 1e-6
MAX_WAV_VALUE = 32768.0

def audioread(path, sampling_rate):
    data, fs = sf.read(path)    
    if fs != sampling_rate:
        data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
    if len(data.shape) >1:
        data = data[:, 0]    
    return  data

def audioread_multi_wavs(paths, sampling_rate):
    data_list = []
    for path in paths:
        data, fs = sf.read(path)
        if fs != sampling_rate:
            data = librosa.resample(data, orig_sr=fs, target_sr=sampling_rate)
        if len(data.shape) >1:
            data = data[:, 0]
        data_list.append(data)
    max_len = 0
    for data in data_list:
        if max_len < len(data):
            max_len = len(data)

    data_list_padded = []
    for data in data_list:
        if max_len > len(data):
            data = zero_padding(data, max_len)
        data_list_padded.append(data)
    return  np.transpose(np.asarray(data_list_padded))

def zero_padding(samples, desired_length):
    samples_to_add = desired_length - len(samples)
    if len(samples.shape) == 1:
        new_zeros = np.zeros(samples_to_add)
    elif len(samples.shape) == 2:
        new_zeros = np.zeros((samples_to_add, samples.shape[1]))
    return np.append(samples, new_zeros, axis=0)

def audio_norm(x):
    rms = (x ** 2).mean() ** 0.5
    scalar = 10 ** (-25 / 20) / (rms + EPS)
    x = x * scalar
    pow_x = x**2
    avg_pow_x = pow_x.mean()
    rmsx = pow_x[pow_x>avg_pow_x].mean()**0.5
    scalarx = 10 ** (-25 / 20) / (rmsx + EPS)
    x = x * scalarx
    return x

class DataReader(object):
    def __init__(self, args):
        self.file_list = read_and_config_file(args.input_path, args, decode=True)
        self.sampling_rate = args.sampling_rate
    def extract_feature(self, path):
        utt_id = path.split('/')[-1]
        data = audioread(path, self.sampling_rate).astype(np.float32)
        inputs = np.reshape(data, [1, data.shape[0]])
        return inputs, utt_id, data.shape[0]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        return self.extract_feature(self.file_list[index])

class Wave_Processor(object):
    def process_two_wavs(self, path, segment_length, sampling_rate):         
        try:
            wave_inputs = audioread(path['inputs'], sampling_rate)
            wave_labels = audioread(path['labels'], sampling_rate)
        except:
            print('audioread() failed, skip this batch!')
            return None, None
        len_wav = wave_labels.shape[0]
        if wave_inputs.shape[0] < segment_length:
            padded_inputs = np.zeros(segment_length, dtype=np.float32)
            padded_labels = np.zeros(segment_length, dtype=np.float32)
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_labels[:wave_labels.shape[0]] = wave_labels
        else:
            st_idx = random.randint(0, len_wav - segment_length)
            padded_inputs = wave_inputs[st_idx:st_idx+segment_length]
            padded_labels = wave_labels[st_idx:st_idx+segment_length]
        return padded_inputs, padded_labels

    def process_multi_labels(self, path, segment_length, sampling_rate):
        try:
            wave_inputs = audioread(path['inputs'], sampling_rate)        
            wave_labels = audioread_multi_wavs(path['labels'], sampling_rate)
        except:
            print('audioread() failed, skip this batch!')
            return None, None
        max_len = max(wave_inputs.shape[0], wave_labels.shape[0])
        if max_len < segment_length:
            padded_inputs = zero_padding(wave_inputs, segment_length)
            padded_labels = zero_padding(wave_labels, segment_length)
            padded_inputs[:wave_inputs.shape[0]] = wave_inputs
            padded_labels[:wave_labels.shape[0],:] = wave_labels
        else:
            st_idx = random.randint(0, max_len - segment_length)
            padded_inputs = wave_inputs[st_idx:st_idx+segment_length]
            padded_labels = wave_labels[st_idx:st_idx+segment_length,:]
        return padded_inputs, padded_labels

    def process_fbanks(self, inputs, args):
        frame_length = int(args.win_len/args.sampling_rate * 1000)
        frame_shift = int(args.win_inc/args.sampling_rate * 1000)

        fbank_config={
        "dither":1.0,
        "frame_length": frame_length,
        "frame_shift": frame_shift,
        "num_mel_bins": args.num_mels, 
        "sample_frequency": args.sampling_rate, 
        "window_type": args.win_type
        }

        inputs = torch.FloatTensor(inputs * MAX_WAV_VALUE)
        fbank = torchaudio.compliance.kaldi.fbank(inputs.unsqueeze(0), **fbank_config)
        ##add delta and delta-delta
        fbank_tr = torch.transpose(fbank, 0, 1)
        fbank_delta = torchaudio.functional.compute_deltas(fbank_tr)
        fbank_delta_delta = torchaudio.functional.compute_deltas(fbank_delta)
        fbank_delta = torch.transpose(fbank_delta, 0, 1)
        fbank_delta_delta = torch.transpose(fbank_delta_delta, 0, 1)
        fbanks = torch.cat([fbank, fbank_delta, fbank_delta_delta], dim=1)               
        return fbanks.numpy()

class AudioDataset(Dataset):

    def __init__(
            self,
            args,
            data_type,
        ):
        self.args = args
        self.sampling_rate = args.sampling_rate
        if data_type == 'train':
            self.wav_list = read_and_config_file(args.tr_list, args)
        elif data_type == 'val':
            self.wav_list = read_and_config_file(args.cv_list, args)
        elif data_type == 'test':
            self.wav_list = read_and_config_file(args.tt_list, args)
        else:
            print(f'Data type: {data_type} is unknown!')
        self.wav_processor = Wave_Processor()
        self.segment_length = self.sampling_rate*self.args.max_length 
        print(f'No. {data_type} files: {len(self.wav_list)}')

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        data_info = self.wav_list[index]
        while 1:
            if self.args.load_type == 'one_input_one_output':
                inputs, labels = self.wav_processor.process_two_wavs({'inputs':data_info['inputs'],'labels':data_info['labels']}, self.segment_length, self.sampling_rate)
            elif self.args.load_type == 'one_input_multi_outputs':
                inputs, labels = self.wav_processor.process_multi_labels({'inputs':data_info['inputs'],'labels':data_info['labels']}, self.segment_length, self.sampling_rate)
            else:
                print(f'data loading for {self.args.load_type} is not implemented!')
                break
                return
            if inputs is not None:
                break
            else:
                index = index + 1
                if index >= len(self.wav_list): index = random.randint(0, len(self.wav_list)-1)
                data_info = self.wav_list[index]        
        return inputs, labels 

def zero_pad_concat(self, inputs):
    max_t = max(inp.shape[0] for inp in inputs)
    shape=None
    if len(inputs[0].shape) == 1:
        shape = (len(inputs), max_t)
    elif len(inputs[0].shape) == 2:
        shape = (len(inputs), max_t,inputs[0].shape[1])
    input_mat = np.zeros(shape, dtype=np.float32)
    for e, inp in enumerate(inputs):
        if len(inp.shape) == 1:
            input_mat[e,:inp.shape[0]] = inp 
        elif len(inp.shape) == 2:
            input_mat[e,:inp.shape[0],:] = inp
    return input_mat

def collate_fn_2x_wavs(data):
    inputs, labels = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    return x, y

def collate_fn_2x_wavs_fbank(data):
    inputs, labels, fbanks = zip(*data)
    x = torch.FloatTensor(inputs)
    y = torch.FloatTensor(labels)
    z = torch.FloatTensor(fbanks)
    return x, y, z 

class DistributedSampler(data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            ind = torch.randperm(int(len(self.dataset)/self.num_replicas), generator=g)*self.num_replicas
            indices = []
            for i in range(self.num_replicas):
                indices = indices + (ind+i).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        indices = indices[self.rank*self.num_samples:(self.rank+1)*self.num_samples]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_dataloader(args, data_type):
    datasets = AudioDataset(
                args=args,
                data_type=data_type)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None
    '''
    if args.network in ['MossFormer2_SS_16K', 'MossFormer2_SS_8K'] :
        collate_fn = collate_fn_2x_wavs
    else:
        print(f'in dataloader, please specify a correct network type using args.network!')
        return
    '''
    # we use collate_fn_2x_wavs as default data collection func
    collate_fn = collate_fn_2x_wavs
    
    generator = data.DataLoader(datasets,
            batch_size = args.batch_size,
            shuffle = (sampler is None),
            collate_fn=collate_fn,
            num_workers = args.num_workers,
            sampler=sampler)
    return sampler, generator
    
