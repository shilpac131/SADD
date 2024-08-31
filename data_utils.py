import os
import random
import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"

def genSpoof_list_spk(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    spk_id = {}
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            spk, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
            spk_id[key] = spk
        return d_meta, file_list, spk_id


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x

## for sSADD network

class Dataset_ASVspoof2019_sSADD(Dataset):
    def __init__(self, list_IDs, labels, spk_IDs, base_dir, LPC_dir,set_type):
        """self.list_IDs    : list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""

        self.list_IDs = list_IDs
        self.labels = labels
        self.spk_IDs = spk_IDs
        self.base_dir = base_dir
        self.set_type = set_type
        self.LPC_dir = LPC_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        key = self.list_IDs[index]
        spk = self.spk_IDs[key]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_test = Tensor(X_pad)
        y = self.labels[key]

        ## for LPC res wav X_test_res
        X_res,_ = sf.read(str(self.LPC_dir / f"flac/{key}.wav"))
        X_pad_res = pad_random(X_res, self.cut)
        x_test_res = Tensor(X_pad_res)

        # for LPC res wav files
        spk_path = "path" # path to your spk embedding extracted from LPC analysis
        wav_files = [file for file in os.listdir(spk_path) if file.endswith(".wav")]

        # Choose a random WAV file
        random_wav_file = random.choice(wav_files)
        random_wav_path = os.path.join(spk_path, random_wav_file)

        LPC_res,_ = sf.read(random_wav_path)
        LPC_res_pad = pad(LPC_res, self.cut)
        x_enroll_res = Tensor(LPC_res_pad)
        
        return x_test, x_test_res, x_enroll_res, y

## for SADD network

class Dataset_ASVspoof2019_SADD(Dataset):
    def __init__(self, list_IDs, labels, spk_IDs, base_dir, set_type):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""

        self.list_IDs = list_IDs
        self.labels = labels
        self.spk_IDs = spk_IDs
        self.base_dir = base_dir
        self.set_type = set_type
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):

        key = self.list_IDs[index]
        spk = self.spk_IDs[key]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_test = Tensor(X_pad)
        y = self.labels[key]

        # for LPC res wav files
        
        spk_path = "path" # path to your spk embedding extracted from LPC analysis
        wav_files = [file for file in os.listdir(spk_path) if file.endswith(".wav")]
        
        # Choose a random WAV file
        random_wav_file = random.choice(wav_files)
        random_wav_path = os.path.join(spk_path, random_wav_file)

        LPC_res,_ = sf.read(random_wav_path)
        LPC_res_pad = pad_random(LPC_res, self.cut)
        x_enroll_res = Tensor(LPC_res_pad)
        return x_test, y, x_enroll_res

