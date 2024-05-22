import librosa
import librosa.display
import numpy as np
from loguru import logger
import noisereduce as nr


def count_similar_sounds(target_audio_path, recording_path, similarity_threshold):
    try:
        # Load the target sound and recording
        logger.info(
            "Target audio path: {}", target_audio_path
        )  # Log the target audio path
        logger.info("Recording path: {}", recording_path)  # Log the recording path
        target_audio, target_audio_sr = librosa.load(target_audio_path, sr=None)  # Load the target audio file
        reduced_noise_sound = nr.reduce_noise(y=target_audio, sr=int(target_audio_sr))
        recorded_audio, recorded_audio_sr = librosa.load(recording_path, sr=None)  # Load the recording audio file
        reduced_noise_recorded = nr.reduce_noise(y=recorded_audio, sr=int(recorded_audio_sr))
        # Initialize a count variable to track the number of occurrences
        count = 0

        # Iterate over the recording
        for i in range(len(reduced_noise_recorded) - len(reduced_noise_sound)):
            audio_segment = reduced_noise_recorded[
                            i: i + len(reduced_noise_sound)
                            ]  # Extract a segment of audio

            # Calculate the similarity score between the audio segment and the target audio
            similarity_score = np.dot(audio_segment, reduced_noise_sound) / (
                    np.linalg.norm(audio_segment) * np.linalg.norm(reduced_noise_sound)
            )

            # Check if the similarity score exceeds the threshold
            if similarity_score >= similarity_threshold:
                count += 1  # Increment the count if similarity score is above threshold

        # Return the total number of occurrences
        return count

    except Exception as e:
        logger.exception("Oops", e)  # Log any exceptions that occur
        return 0  # Return 0 in case of an exception


def main():
    sound_path = "us_short.ogg"
    recording_path = "us1259_process_long.wav"
    print(
        "number of acourences: "
        + str(count_similar_sounds(sound_path, recording_path, 0.27))
    )


if __name__ == "__main__":
    main()
