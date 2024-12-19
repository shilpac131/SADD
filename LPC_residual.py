import scipy as sp
import argparse
import numpy as np
import matplotlib.pyplot as plt
import librosa as lb
import scipy.signal as sg
from scipy.io import wavfile
import scipy.io.wavfile
from scipy.signal.windows import hann, hamming
from scipy import fft
import math
from math import floor
import os
import soundfile as sf
import librosa
import librosa.display
from tqdm import tqdm


# split the frames
def split_frames(signal, window, overlap=0.5):
    
    n = len(signal) #length of input
    nw = len(window) #window size
    step = math.floor(nw*(1-overlap)) 
    
    nf = math.floor((n-nw)/step)+1 #number of frames
    
    frames = np.zeros((nf, nw))
    
    for i in range(nf):
        offset = i*step        
        frames[i, :] = window*signal[offset: nw+offset]
    
    return frames

def AddFrameBlocks(S_frames, window, Overlap = 0.5):
    f_count, nw = S_frames.shape
    # print('Frame count: {}, Window length: {}'.format(f_count, nw))
    step = np.floor(nw * Overlap)
    # print('Step size: ', step)

    n = (f_count-1) * step + nw
    x = np.zeros((int(n), ))

    for i in range(f_count):
        offset = int(i * step)
        x[offset : nw + offset] += S_frames[i, :]

    return x

def residual(frames, order):
    # Iterate over all frames for LPC analysis
    p = order
    x_hat = []
    residual = []
    for frame in frames:
        coeff = lb.lpc(frame, order = p)
        res = sg.lfilter(coeff,[1],frame)
        syn = sg.lfilter([1], coeff, res)
        x_hat.append(syn)
        residual.append(res)
    return np.asarray(x_hat), np.asarray(residual)

def plotLPC(frames, frame_no, window, p,sr):
    frame = frames[frame_no]
    
    NFFT = fft.next_fast_len(window,True)
    freqAxis = fft.fftfreq(NFFT,d=1/sr)
    X = np.log(np.abs(sp.fft.fft(frame,n=NFFT)))
    # since the signal is real, we need only half the mag spectrum
    X = X[0:NFFT//2]
    #LPC Spectrum
    a = lb.lpc(frame, order = p)
    res = sg.lfilter(a, [1], frame)
    g = np.sqrt(np.sum(res**2))
    w, h = sg.freqz(g, a, worN=NFFT//2)

    # plot with freq in pi units
    freqAxis2 = (np.pi/(NFFT//2))*(np.linspace(0,(NFFT//2)-1,num=NFFT//2))
    plt.figure()
    plt.plot(freqAxis2/np.pi,X)
    plt.plot(freqAxis2/np.pi,np.log(np.abs(h)))
    plt.grid(True)
    plt.xlabel('Freq in pi units')
    plt.ylabel('Log mag spectrum')
    plt.legend(['DFT spec','LPC spec'])
    plt.show()

    return a



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LPC calculation")
    parser.add_argument('--source', type=str,
                        default='/home/shilpa/Desktop/LPC/oak.wav', help='add source file path')
    parser.add_argument('--save_file', action='store_true', default=False,help='if u want to store the reconstructed aud file')
    parser.add_argument('--order', type=int,
                        default=16, help='Pls input order value')
    parser.add_argument('--set_type', type=str,
                        default='train', help='Pls input set type :{train,dev,eval}')
    args = parser.parse_args()

    set = args.set_type
    # Input and output folders
    input_folder = f"./LA/ASVspoof2019_LA_{set}/flac"
    output_folder = f"./LPC_Res_wavs/ASVspoof2019_LA_{set}/flac"


    sym = False
    
    print(f"for {set} set in LA partition")
    # Loop through all files in the input folder
    for file_name in tqdm(os.listdir(input_folder)):
        input_file_path = os.path.join(input_folder, file_name)
        # read data
        data, fs = sf.read(input_file_path)
        window = hamming(math.floor(0.03*fs), sym)
        frames = split_frames(data, window, 0.5)

        x_hat, residue  = residual(frames, order = args.order)
        final_residual = AddFrameBlocks(residue, window)
        file_name_without_extension = os.path.splitext(file_name)[0]
        output_file_path = os.path.join(output_folder, file_name_without_extension)
        # np.save(f'{output_file_path}.npy', final_residual)
        sf.write(f'{output_file_path}.wav', final_residual, fs)

        # print(f"residual for {file_name} is calculated and saved")

        # print(f"no of samples in original audio: {len(data)} \nno of samples in residual: {len(final_residual)}")












