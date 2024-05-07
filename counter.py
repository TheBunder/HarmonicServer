import librosa
import librosa.display
import numpy as np
from loguru import logger


def count_similar_sounds(target_audio_path, recording_path, similarity_threshold):
    try:
        # Load the target sound and recording
        logger.info(
            "Target audio path: {}", target_audio_path
        )  # Log the target audio path
        logger.info("Recording path: {}", recording_path)  # Log the recording path
        target_audio, _ = librosa.load(target_audio_path)  # Load the target audio file
        recorded_audio, _ = librosa.load(
            recording_path
        )  # Load the recording audio file

        # Initialize a count variable to track the number of occurrences
        count = 0

        # Iterate over the recording
        for i in range(len(recorded_audio) - len(target_audio)):
            audio_segment = recorded_audio[
                i : i + len(target_audio)
            ]  # Extract a segment of audio

            # Calculate the similarity score between the audio segment and the target audio
            similarity_score = np.dot(audio_segment, target_audio) / (
                np.linalg.norm(audio_segment) * np.linalg.norm(target_audio)
            )

            # Check if the similarity score exceeds the threshold
            if similarity_score >= similarity_threshold:
                count += 1  # Increment the count if similarity score is above threshold

        # Return the total number of occurrences
        return count

    except Exception as e:
        logger.exception("Oops", e)  # Log any exceptions that occur
        return 0  # Return 0 in case of an exception
