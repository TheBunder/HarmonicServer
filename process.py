import datetime
import os
import re
import sys

import librosa
import numpy as np
import scipy
import scipy.signal
from librosa import ParameterError
from loguru import logger
from matplotlib import pyplot as plt
from typing import Final

DTYPE = np.complex64
N_FFT: Final[int] = 2048
HZ_COUNT: Final[int] = int(1 + N_FFT // 2)  # 1025 (Hz buckets)
WIN_LENGTH: Final[int] = N_FFT
HOP_LENGTH: Final[int] = int(WIN_LENGTH // 4)
SAMPLE_RATE: Final[int] = 44100
# sample_crop_start = 5  # The first 4 seem to get damaged
# sample_crop_end = 4
SAMPLE_WARN_ALLOWANCE: Final[int] = 3

MATCH_ANY_SAMPLE: Final[bool] = True


# --------------------------------------------------
def stft_raw(series, sample_rate, win_length, hop_length, hz_count, dtype):
    # --------------------------------------------------
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

    lpad = int((N_FFT - n) // 2)

    lengths = [(0, 0)] * fft_window.ndim
    lengths[axis] = (lpad, int(N_FFT - n - lpad))

    if lpad < 0:
        raise ParameterError(('Target size ({:d}) must be at least input size ({:d})').format(N_FFT, n))

    fft_window = np.pad(fft_window, lengths, mode='constant')

    # --------------------------------------------------
    # Reshape so that the window can be broadcast

    fft_window = fft_window.reshape((-1, 1))

    # --------------------------------------------------
    # Pad the time series so that frames are centred

    series = np.pad(series, int(N_FFT // 2), mode=pad_mode)

    # --------------------------------------------------
    # Window the time series.

    # Compute the number of frames that will fit. The end may get truncated.
    frame_count = 1 + int((len(series) - N_FFT) / hop_length)  # Where n_fft = frame_length

    # Vertical stride is one sample
    # Horizontal stride is `hop_length` samples
    frames_data = np.lib.stride_tricks.as_strided(series, shape=(N_FFT, frame_count),
                                                  strides=(series.itemsize, hop_length * series.itemsize))

    # --------------------------------------------------
    # how many columns can we fit within MAX_MEM_BLOCK

    MAX_MEM_BLOCK = 2 ** 8 * 2 ** 10
    n_columns = int(MAX_MEM_BLOCK / (hz_count * dtype(0).itemsize))

    # --------------------------------------------------
    # Return

    return frames_data, fft_window, n_columns


# --------------------------------------------------

config = {

    'source_path': os.path.join('source', 'us1382_process_long.wav'),
    'source_frame_start': 0,  # (x * sample_rate) / hop_length)
    'source_frame_end': None,  # (x * sample_rate) / hop_length)

    'matching_samples': os.path.join('samples', 'us_short.wav'),
    'matching_min_score': 0.16,
    'matching_skip': 0,  # Jump forward X seconds after a match.
    'matching_ignore': 0,  # Ignore additional matches X seconds after the last one.

    'output_title': None,  # Set a title to create ".meta" file, and "X-chapters.mp3"

}


def analyze_sound(sample_path):
    # Config

    # sample_path = sample_info[0]
    sample_crop_start = 5  # The first 4 seem to get damaged
    sample_crop_end = 4

    sample_path_split = os.path.split(sample_path)
    sample_ext_split = os.path.splitext(sample_path_split[1])

    # series_data = sample_info[1]
    series_data = librosa.load(sample_path, sr=None)
    series_max_length = series_data[0].shape[0]

    # --------------------------------------------------
    # Original frame length

    stft_frames, fft_window, n_columns = stft_raw(series_data[0], SAMPLE_RATE, WIN_LENGTH, HOP_LENGTH, HZ_COUNT, DTYPE)

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

    stft_frames, fft_window, n_columns = stft_raw(series_data[0], SAMPLE_RATE, WIN_LENGTH, HOP_LENGTH, HZ_COUNT, DTYPE)

    # Pre-allocate the STFT matrix
    stft_data = np.empty((int(1 + N_FFT // 2), stft_frames.shape[1]), dtype=DTYPE, order='F')

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

    stft_crop_start_time = ((float(stft_crop_start) * HOP_LENGTH) / SAMPLE_RATE)
    stft_crop_end_time = ((float(stft_crop_end) * HOP_LENGTH) / SAMPLE_RATE)

    # --------------------------------------------------
    # Plot

    plt.figure(figsize=(5, 6))

    plt.subplot(2, 1, 1)
    librosa.display.waveshow(series_harm, sr=SAMPLE_RATE, alpha=0.25)
    librosa.display.waveshow(series_perc, sr=SAMPLE_RATE, color='r', alpha=0.5)
    plt.axvline(x=stft_crop_start_time)
    plt.axvline(x=stft_crop_end_time)
    plt.tight_layout()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(stft_data, sr=SAMPLE_RATE, x_axis='time', y_axis='log', cmap='Reds')
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

    # logger.info('{} ({}/{}) - {}'.format(sample_path, stft_crop_start, stft_crop_end, series_length))


def counter(source_path, matching_sample):
    sample_crop_start = 5  # The first 4 seem to get damaged
    sample_crop_end = 4

    config['source_path'] = source_path
    config['matching_samples'] = matching_sample

    logger.info('Counting ' + str(config['matching_samples']) + ' in ' + str(config['source_path']))

    logger.info('Config')
    # logger.info('Hz Min Score: {}'.format(config['matching_min_score']))

    cnt = 0

    # --------------------------------------------------

    start_time = datetime.datetime.now()

    # --------------------------------------------------

    logger.info('Load Source')

    if not os.path.exists(config['source_path']):
        logger.info('Missing source file')
        sys.exit()

    source_series, source_sr = librosa.load(config['source_path'], sr=None)

    source_time_total = (float(len(source_series)) / source_sr)

    # logger.info('{} ({} & {})'.format(config['source_path'], source_time_total, source_sr))

    # --------------------------------------------------

    logger.info('Load Samples')

    samples = []

    if not os.path.exists(config['matching_samples']):
        logger.info('Missing samples folder: ' + config['matching_samples'])
        sys.exit()

    if os.path.isdir(config['matching_samples']):
        files = []
        for path in os.listdir(config['matching_samples']):
            path = os.path.join(config['matching_samples'], path)
            if os.path.isfile(path) and not os.path.basename(path).startswith('.'):
                files.append(path)
        files = sorted(files)
    else:
        files = [config['matching_samples']]

    for sample_path in files:
        if os.path.isfile(sample_path):

            sample_series, sample_sr = librosa.load(sample_path, sr=None)  # load the sound from file

            sample_frames, fft_window, n_columns = stft_raw(sample_series, sample_sr, WIN_LENGTH, HOP_LENGTH, HZ_COUNT,
                                                            DTYPE)

            # Pre-allocate the STFT matrix
            sample_data = np.empty((int(1 + N_FFT // 2), sample_frames.shape[1]), dtype=DTYPE, order='F')

            for bl_s in range(0, sample_data.shape[1], n_columns):  # process the data
                bl_t = min(bl_s + n_columns, sample_data.shape[1])
                sample_data[:, bl_s:bl_t] = scipy.fft.fft(fft_window * sample_frames[:, bl_s:bl_t], axis=0)[
                                            :sample_data.shape[0]]

            sample_data = abs(sample_data)

            sample_height = sample_data.shape[0]
            sample_length = sample_data.shape[1]

            x = 0
            sample_start = 0
            while x < sample_length:
                total = 0
                for y in range(0, sample_height):
                    total += sample_data[y][x]
                if total >= 1:
                    sample_start = x
                    break
                x += 1
            sample_start += sample_crop_start  # The first few frames seem to get modified, perhaps due to compression?
            sample_end = (sample_length - sample_crop_end)

            samples.append([
                sample_start,
                sample_end,
                os.path.basename(sample_path),
                sample_data
            ])

            # logger.info('  {} ({}/{})'.format(sample_path, sample_start, sample_end))

    # --------------------------------------------------
    # Processing

    logger.info('Processing')

    source_frames, fft_window, n_columns = stft_raw(source_series, SAMPLE_RATE, WIN_LENGTH, HOP_LENGTH, HZ_COUNT, DTYPE)

    if config['source_frame_end'] is None:
        config['source_frame_end'] = source_frames.shape[1]

    # logger.info('From {} to {}'.format(config['source_frame_start'], config['source_frame_end']))
    # logger.info('From {} to {}'.format(((float(config['source_frame_start']) * hop_length) / sample_rate),
    #                                    ((float(config['source_frame_end']) * hop_length) / sample_rate)))

    matching = {}
    match_count = 0
    match_last_time = None
    match_skipping = 0
    matches = []

    results_end = {}
    results_dupe = {}
    for sample_id, sample_info in enumerate(samples):
        results_end[sample_id] = {}
        results_dupe[sample_id] = {}
        for k in range(0, (sample_info[1] + 1)):
            results_end[sample_id][k] = 0
            results_dupe[sample_id][k] = 0

    for block_start in range(config['source_frame_start'], config['source_frame_end'], n_columns):  # Time in 31 blocks

        block_end = min(block_start + n_columns, config['source_frame_end'])

        set_data = abs((scipy.fft.fft(fft_window * source_frames[:, block_start:block_end], axis=0)).astype(DTYPE))

        # logger.info('{} to {} - {}'.format(block_start, block_end, str(datetime.timedelta(
        #     seconds=((float(block_start) * hop_length) / sample_rate)))))

        x: int = 0
        x_max = (block_end - block_start)
        while x < x_max:

            if match_skipping > 0:
                if x == 0:
                    logger.info('Skipping {}'.format(match_skipping))
                match_skipping -= 1
                x += 1
                continue

            matching_complete = []
            for matching_id in list(matching):  # Continue to check matches (i.e. have already started)

                sample_id = matching[matching_id][0]
                sample_x = (matching[matching_id][1] + 1)

                if sample_id in matching_complete:
                    continue
                dimensions = set_data.shape
                if dimensions[1] == 0:
                    for match in matches:
                        cnt += 1
                    return cnt
                hz_score = abs(
                    set_data[0:HZ_COUNT, max(0, min(x, dimensions[1] - 1))] - samples[sample_id][3][0:HZ_COUNT, sample_x])
                hz_score = sum(hz_score) / float(len(hz_score))  # calculate similarity

                if hz_score < config['matching_min_score']:  # Is it above or below the minimum to be count

                    if sample_x >= samples[sample_id][1]:

                        match_start_time = ((float(x + block_start - samples[sample_id][1]) * HOP_LENGTH) / SAMPLE_RATE)

                        logger.info(
                            'Match {}/{}: Complete at {} @ {}'.format(matching_id, sample_id, sample_x,
                                                                      match_start_time))

                        results_end[sample_id][sample_x] += 1

                        if (config['matching_skip']) or (match_last_time is None) or (
                                (match_start_time - match_last_time) > config['matching_ignore']):
                            match_last_ignored = False
                        else:
                            match_last_ignored = True

                        matches.append([sample_id, match_start_time, match_last_ignored])  # Add the match to the list
                        match_last_time = match_start_time

                        if config['matching_skip']:
                            match_skipping = ((config['matching_skip'] * SAMPLE_RATE) / HOP_LENGTH)
                            logger.info('Skipping {}'.format(match_skipping))
                            matching = {}
                            break  # No more 'matching' entires
                        else:
                            del matching[matching_id]
                            matching_complete.append(sample_id)

                    else:

                        # logger.info(
                        #     'Match {}/{}: Update to {} ({} < {})'.format(matching_id, sample_id, sample_x, hz_score,
                        #                                                  config['matching_min_score']))
                        matching[matching_id][1] = sample_x

                elif matching[matching_id][2] < SAMPLE_WARN_ALLOWANCE and sample_x > 10:

                    # logger.info('Match {}/{}: Warned at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x,
                    #                                                                samples[sample_id][1], hz_score,
                    #                                                                config['matching_min_score']))
                    matching[matching_id][2] += 1

                else:

                    # logger.info('Match {}/{}: Failed at {} of {} ({} > {})'.format(matching_id, sample_id, sample_x,
                    #                                                                samples[sample_id][1], hz_score,
                    #                                                                config['matching_min_score']))
                    results_end[sample_id][sample_x] += 1
                    del matching[matching_id]

            if match_skipping > 0:
                continue

            for matching_sample_id in matching_complete:
                for matching_id in list(matching):
                    if MATCH_ANY_SAMPLE or matching[matching_id][0] == matching_sample_id:
                        sample_id = matching[matching_id][0]
                        sample_x = matching[matching_id][1]
                        # logger.info('Match {}/{}: Duplicate Complete at {}'.format(matching_id, sample_id, sample_x))
                        results_dupe[sample_id][sample_x] += 1
                        del matching[
                            matching_id]  # Cannot be done in the first loop (next to continue), as the order in a dictionary is undefined, so you could have a match that started later, getting tested first.

            for sample_id, sample_info in enumerate(
                    samples):  # For each sample, see if the first frame (after sample_crop_start), matches well enough to keep checking (that part is done above).

                sample_start = sample_info[0]

                # TEST-1
                dimensions = set_data.shape
                if dimensions[1] == 0:
                    for match in matches:
                        cnt += 1
                    return cnt
                hz_score = abs(
                    set_data[0:HZ_COUNT, max(0, min(x, dimensions[1] - 1))] - sample_info[3][0:HZ_COUNT, sample_start])
                hz_score = sum(hz_score) / float(len(hz_score))

                if hz_score < config['matching_min_score']:
                    match_count += 1
                    # logger.info('Match {}: Start for sample {} at {} ({} < {})'.format(match_count, sample_id,(x + block_start),hz_score, config['matching_min_score']))
                    matching[match_count] = [
                        sample_id,
                        sample_start,
                        0,  # Warnings
                    ]

            x += 1

    # --------------------------------------------------

    logger.info('Matches')
    for match in matches:
        cnt += 1
        logger.info(' {} = {} @ {}{}'.format(samples[match[0]][2], str(datetime.timedelta(seconds=match[1])), match[1],
                                             (' - Ignored' if match[2] else '')))

    # --------------------------------------------------
    logger.info(datetime.datetime.now() - start_time)
    return cnt


def main():
    counter(os.path.join('us_sources', 'us8066_process_long.wav'), os.path.join('us_sounds', 'us_short.wav'))


if __name__ == '__main__':
    main()
