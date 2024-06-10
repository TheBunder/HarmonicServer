# --------------------------------------------------

import glob
import os
import re
import subprocess
import sys

from loguru import logger

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
from librosa import ParameterError


# --------------------------------------------------
def stft_raw(series, sample_rate, win_length, hop_length, hz_count, dtype):
    # Config

    window = 'hann'
    pad_mode = 'reflect'

    # --------------------------------------------------
    # Get Window

    fft_window = scipy.signal.get_window(window, win_length, fftbins=True)

    # --------------------------------------------------
    # Pad the window out to n_fft size... Wrapper for
    # np.pad to automatically centre an array prior to padding.

    axis = -1

    n = fft_window.shape[axis]

    lpad = int((n_fft - n) // 2)

    lengths = [(0, 0)] * fft_window.ndim
    lengths[axis] = (lpad, int(n_fft - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be at least input size ({:d})').format(n_fft, n))

    fft_window = np.pad(fft_window, lengths, mode='constant')

    # --------------------------------------------------
    # Reshape so that the window can be broadcast

    fft_window = fft_window.reshape((-1, 1))

    # --------------------------------------------------
    # Pad the time series so that frames are centred

    series = np.pad(series, int(n_fft // 2), mode=pad_mode)

    # --------------------------------------------------
    # Window the time series.

    # Compute the number of frames that will fit. The end may get truncated.
    frame_count = 1 + int((len(series) - n_fft) / hop_length)  # Where n_fft = frame_length

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    frames_data = np.lib.stride_tricks.as_strided(series, shape=(n_fft, frame_count),
                                                  strides=(series.itemsize, hop_length * series.itemsize))

    # --------------------------------------------------
    # how many columns can we fit within MAX_MEM_BLOCK

    MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
    n_columns = int(MAX_MEM_BLOCK / (hz_count * dtype(0).itemsize))

    # --------------------------------------------------
    # Return

    return frames_data, fft_window, n_columns


# --------------------------------------------------

dtype = np.complex64
n_fft = 2048
hz_count = int(1 + n_fft // 2)  # 1025 (Hz buckets)
win_length = n_fft
hop_length = int(win_length // 4)
sample_rate = 44100
sample_crop_start = 5  # The first 4 seem to get damaged
sample_crop_end = 4
sample_warn_allowance = 3

match_any_sample = True
# --------------------------------------------------

sample_path = r"samples\us_short.wav"
sample_path_split = os.path.split(sample_path)
sample_ext_split = os.path.splitext(sample_path_split[1])

# --------------------------------------------------

samples_folder = 'samples'

if not os.path.exists(samples_folder):
    logger.info('Missing samples folder: ' + samples_folder)
    sys.exit()

# --------------------------------------------------

samples = []
series_max_length = 0

if os.path.isdir(samples_folder):
    files = sorted(glob.glob(os.path.join(samples_folder, '*')))
else:
    files = [samples_folder]

for sample_path in files:
    if os.path.isfile(sample_path):

        series_data = librosa.load(sample_path, sr=None)
        # what is shape
        if series_max_length < series_data[0].shape[0]:
            series_max_length = series_data[0].shape[0]

        samples.append([
            sample_path,
            series_data,
        ])

# --------------------------------------------------


logger.info(series_max_length)


# for sample_id, sample_info in enumerate(samples):
def analyze_sound(sample_path):
    # Config

    # sample_path = sample_info[0]
    sample_path_split = os.path.split(sample_path)
    sample_ext_split = os.path.splitext(sample_path_split[1])

    # series_data = sample_info[1]
    series_data = librosa.load(sample_path, sr=None)
    series_max_length = series_data[0].shape[0]

    # --------------------------------------------------
    # Original frame length

    stft_frames, fft_window, n_columns = stft_raw(series_data[0], sample_rate, win_length, hop_length, hz_count, dtype)

    stft_length_source = stft_frames.shape[1]

    # --------------------------------------------------
    # All samples the same length

    series_length = series_data[0].shape[0]
    if series_max_length > series_length:
        series_padding = np.full((series_max_length - series_length), 0)
        series_data = np.concatenate((series_data, series_padding), axis=0)

    # --------------------------------------------------
    # Harmonic and percussive components

    series_harm, series_perc = librosa.effects.hpss(series_data[0])

    # --------------------------------------------------
    # STFT data

    stft_frames, fft_window, n_columns = stft_raw(series_data[0], sample_rate, win_length, hop_length, hz_count, dtype)

    # Pre-allocate the STFT matrix
    stft_data = np.empty((int(1 + n_fft // 2), stft_frames.shape[1]), dtype=dtype, order='F')

    for bl_s in range(0, stft_data.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_data.shape[1])
        stft_data[:, bl_s:bl_t] = scipy.fft.fft(fft_window * stft_frames[:, bl_s:bl_t], axis=0)[:stft_data.shape[0]]

    stft_data = abs(stft_data)

    stft_height = stft_data.shape[0]
    stft_length_padded = stft_data.shape[1]

    # --------------------------------------------------
    # Start

    x = 0
    stft_crop_start = 0
    while x < stft_length_padded:
        total = 0
        for y in range(0, stft_height):
            total += stft_data[y][x]
        if total >= 1:
            stft_crop_start = x
            break
        x += 1
    stft_crop_start += sample_crop_start
    stft_crop_end = (stft_length_source - sample_crop_end)

    stft_crop_start_time = ((float(stft_crop_start) * hop_length) / sample_rate)
    stft_crop_end_time = ((float(stft_crop_end) * hop_length) / sample_rate)

    # --------------------------------------------------
    # Plot

    plt.figure(figsize=(5, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(series_harm, sr=sample_rate, alpha=0.25)
    librosa.display.waveshow(series_perc, sr=sample_rate, color='r', alpha=0.5)
    plt.axvline(x=stft_crop_start_time)
    plt.axvline(x=stft_crop_end_time)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(stft_data, sr=sample_rate, x_axis='time', y_axis='log', cmap='Reds')
    plt.axvline(x=stft_crop_start_time)
    plt.axvline(x=stft_crop_end_time)
    plt.tight_layout()

    fig_dir = os.path.join(sample_path_split[0], 'img')
    os.makedirs(fig_dir, exist_ok=True)
    logger.info('Saving to ' + os.path.dirname(os.path.abspath((os.path.join(fig_dir, sample_ext_split[0] + '.png')))))
    plt.savefig(os.path.join(fig_dir, sample_ext_split[0] + '.png'))

    # --------------------------------------------------
    # Details

    details = {}
    detail_dir = os.path.join('.\\', sample_path_split[0], 'info')
    os.makedirs(detail_dir, exist_ok=True)
    detail_path = os.path.join(detail_dir, sample_ext_split[0] + '.txt')

    if os.path.exists(detail_path):
        p = re.compile('([^:]+): *(.*)')
        f = open(detail_path, 'r')
        for line in f:
            m = p.match(line)
            if m:
                details[m.group(1)] = m.group(2)

    details['crop_start'] = str(stft_crop_start)
    details['crop_end'] = str(stft_crop_end)
    details['length_series'] = str(series_length)

    logger.info('Saving to ' + os.path.dirname(os.path.abspath(detail_path)))
    f = open(detail_path, 'w')
    for field in sorted(iter(details.keys())):
        f.write(field + ': ' + details[field] + '\n')

    # --------------------------------------------------
    # Done

    logger.info('{} ({}/{}) - {}'.format(sample_path, stft_crop_start, stft_crop_end, series_length))


def main():
    analyze_sound(r"samples\us_short.wav")


if __name__ == '__main__':
    main()
