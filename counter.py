import wave
from array import array
from struct import pack
from sys import byteorder

import keyboard
import librosa
import librosa.display
import numpy as np
import pyaudio
from loguru import logger

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100


def count_similar_sounds(target_audio_path, recording_path, similarity_threshold):
    try:
        # Load the target sound and recording
        logger.info("Target audio path: {}", target_audio_path)
        logger.info("Recording path: {}", recording_path)
        target_audio, _ = librosa.load(target_audio_path)
        recorded_audio, _ = librosa.load(recording_path)

        # Initialize a count variable to track the number of occurrences
        count = 0

        # Iterate over the recording
        for i in range(len(recorded_audio) - len(target_audio)):
            audio_segment = recorded_audio[i : i + len(target_audio)]

            # Calculate the similarity score between the audio segment and the target audio
            similarity_score = np.dot(audio_segment, target_audio) / (
                np.linalg.norm(audio_segment) * np.linalg.norm(target_audio)
            )

            # Check if the similarity score exceeds the threshold
            if similarity_score >= similarity_threshold:
                count += 1

        # Print the total number of occurrences
        # print("Number of occurrences:", count)

        # Return the total number of occurrences
        return count

    except Exception as e:
        logger.exception("Oops", e)
        return 0


def is_silent(snd_data):
    """Returns 'True' if below the 'silent' threshold"""
    return max(snd_data) < THRESHOLD


def normalize(snd_data):
    """Average the volume out"""
    MAXIMUM = 16384
    times = float(MAXIMUM) / max(abs(i) for i in snd_data)

    r = array("h")
    for i in snd_data:
        r.append(int(i * times))
    return r


def trim(snd_data):
    """Trim the blank spots at the start and end"""

    def _trim(snd_data):
        snd_started = False
        r = array("h")

        for i in snd_data:
            if not snd_started and abs(i) > THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data


def add_silence(snd_data, seconds):
    """Add silence to the start and end of 'snd_data' of length 'seconds' (float)"""
    silence = [0] * int(seconds * RATE)
    r = array("h", silence)
    r.extend(snd_data)
    r.extend(silence)
    return r


def record():
    """
    Record a word or words from the microphone and
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the
    start and end, and pads with 0.5 seconds of
    blank sound to make sure VLC et al. can play
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    num_silent = 0
    snd_started = False

    r = array("h")

    while 1:
        # little endian, signed short
        snd_data = array("h", stream.read(CHUNK_SIZE))
        if byteorder == "big":
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r


def record_with_key_stop(key_to_stop):
    """
    Record audio from the microphone and stop when a specific key is pressed.

    This function records audio until the specified key is pressed on the keyboard.
    """
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=1,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=CHUNK_SIZE,
    )

    r = array("h")

    while not keyboard.is_pressed(key_to_stop):
        snd_data = array("h", stream.read(CHUNK_SIZE))
        if byteorder == "big":
            snd_data.byteswap()
        r.extend(snd_data)

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)

    return sample_width, r


def record_sound(path):
    """Records from the microphone and outputs the resulting data to 'path'"""
    sample_width, data = record()
    data = pack("<" + ("h" * len(data)), *data)

    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def record_to_file(path, key_to_stop):
    """Records from the microphone and outputs the resulting data to 'path'"""
    sample_width, data = record_with_key_stop(key_to_stop)
    data = pack("<" + ("h" * len(data)), *data)

    wf = wave.open(path, "wb")
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


if __name__ == "__main__":
    target_sound_file = "demo_sound.wav"
    recording_file = "demo_record.wav"
    similarity_threshold = 0.7

    # record the sound
    # record_sound("demo_sound.wav")

    # print("the sound has been record")

    # Record audio until the 'Q' key is pressed
    # key_to_stop_recording = "q"
    # record_to_file("demo_record.wav", key_to_stop_recording)
    try:
        count_similar_sounds("user_short.ogg", "whoosh.wav", similarity_threshold)

        # Ask the user if they want to delete the recorded audio files
        # delete_files = (
        #     input("Do you want to delete the recorded audio files? (y/n): ")
        #     .strip()
        #     .lower()
        # )
        #
        # if delete_files == "y":
        #     # Delete the recorded audio files
        #     try:
        #         os.remove(target_sound_file)
        #         os.remove(recording_file)
        #         print("Recorded audio files deleted.")
        #     except FileNotFoundError:
        #         print("One or both of the files not found.")
        # else:
        #     print("Recorded audio files not deleted.")
    except:
        print("error")
