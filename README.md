# Harmonic - Server (Python)

This repository contains the Python-based server-side code for Harmonic, an application designed to automate the counting of short sound segments within longer recordings. Harmonic simplifies the process of repetitive sound counting, freeing users from manual effort and allowing them to focus on more important tasks. The client-side applications (Android and PC) are available in separate repositories: [Android Client](https://github.com/TheBunder/HarmonicClient), [PC Client](https://github.com/TheBunder/client_harmonic).

Harmonic addresses the common need to count recurring sounds within a longer audio clip. For example, a user can record the sound of a single keyboard key press and then record a longer session of typing. Harmonic will then automatically count how many times that key press sound occurred in the recording. This automation has broad applications, from enhancing sports performance (e.g., counting punches or jump rope repetitions) to research and data analysis.

## Project Details

For a comprehensive overview of the project, please refer to the [Technical Documentation](harmonic_technical_documentation.pdf).

## System Description

Harmonic utilizes a client-server architecture. The server is responsible for:

*   **Receiving Sound Data:** Reliably receives recorded sound data from client applications (Android and PC) using a TCP-based protocol.
*   **Similarity Analysis:** Performs similarity analysis using FFT (Fast Fourier Transform) to identify occurrences of specific sound segments within longer recordings.
*   **Counting Occurrences:** Accurately counts the occurrences of the recorded sound segments.
*   **Response to Client:** Sends the occurrence count back to the user's application for display.
*   **Sound Saving:** Stores short sound files (used for comparison) for future use, managed in a sound library.  Long recording segments are processed and then deleted to optimize storage.
*   **User Authentication:** Manages user accounts and authentication using username and password.

## Communication and Security

*   **Data Transmission Protocol:** Employs a custom TCP-based protocol for secure and reliable data exchange between the client and the server.  The protocol includes message size information, request codes, variables, and data sections to ensure proper handling of different message types.
*   **Encryption:** User information (credentials) is encrypted using AES and Diffie-Hellman encryption to protect against unauthorized access.

## Technologies Used

*   Python
*   `numpy` (for numerical computation)
*   `librosa` (for audio analysis)
*   `scipy` (for signal processing, including FFT)
*   `pycryptodome` (for AES encryption)
*   `sqlite3` (for database interaction)
*   `loguru` (for logging)
*   `socket` (for network communication)
*   `threading` (for handling multiple clients concurrently)
*   `asyncio` (for asynchronous operations)
*   `re` (for regular expressions)
*   `base64` (for encoding/decoding data)
*   `struct` (for packing/unpacking binary data)
*   `os` (for file system operations)
*   `random` (for generating random values)
*   `uuid` (for generating unique identifiers)
*   `datetime` (for time-related operations)
*   `atexit` (for cleanup on exit)
*   `concurrent.futures` (for thread pool management)
*   `sys` (for system-specific parameters)
*   `wave` (for WAV file handling)

## Installation

1. **Prerequisites:**
    * Python installed on your system.
    * Poetry installed on your system. You can install Poetry using:
       ```bash
       pip install poetry
       ```
    * **Important:** The client application and the server *must* be on the same private network (e.g., behind a NAT router). This is essential for communication between the client and server.

2. **Steps:**

    1. Clone the repository:
       ```bash
       git clone [https://github.com/TheBunder/HarmonicServer.git](https://github.com/TheBunder/HarmonicServer.git)
       ```

    2. Navigate to the project directory:
       ```bash
       cd HarmonicServer  # Or the actual name of your project's directory
       ```

    3. Install the project dependencies using Poetry:
       ```bash
       poetry install
       ```
       *(Poetry will automatically create a virtual environment (if one doesn't exist) and install the dependencies listed in your `pyproject.toml` file.)*


## Usage

1.  Run the Harmonic server: `python main.py`

## Communication Protocol Details

The communication between the client and server uses a custom TCP-based protocol. Messages are structured as follows:

**Client Message Format:**

`(Message size)|(Length of message until data)~(Request Code)~(Variable 1)~(Variable 2)~...~(Variable N)~Data`

*   `Message size`: Total size of the message in bytes.
*   `Length of message until data`: Length of the message from the beginning up to (but not including) the `Data` section.
*   `Request Code`:  A code indicating the type of request (e.g., login, recording, analysis).
*   `Variables`: Textual data, such as usernames, passwords, file names, etc., separated by "~".
*   `Data`: The raw audio data (binary format).

**Server Message Format:**

`(Message size)|Data`

*   `Message size`: Total size of the message in bytes.
*   `Data`: The response data (e.g., occurrence count, success/failure messages).

## Sound Storage

*   Short sound files (used for comparison) are stored on the server's file system.  File names are stored in the database for retrieval.
*   Long audio recordings are processed and then deleted to conserve storage space.

## User Authentication

User credentials (username and password) are sent from the client to the server for validation.  The password is hashed using SHA256 before being compared to the stored hash in the database.  Upon successful authentication, the server sends a confirmation message and the list of saved sound file names associated with the user.

## Database Structure

The server uses an SQLite database named `Login.db` to store user information and saved sound files.

**Table: Users**

| Field    | Type   | Description                               |
| -------- | ------ | ----------------------------------------- |
| username | TEXT   | Primary key, unique username for each user |
| salt     | BLOB   | Random salt used for password hashing      |
| password | TEXT   | Hashed password                           |

**Example Values:**

| username   | salt           | password                                     |
| ---------- | -------------- | -------------------------------------------- |
| rick_astley | (binary data) | [Hashed Password]                           |

**Table: SavedSounds**

| Field      | Type   | Description                                                              |
| ---------- | ------ | ------------------------------------------------------------------------ |
| sound_name | TEXT   | Name given to the saved sound by the user                                 |
| file_name  | TEXT   | Name of the audio file stored on the server's file system                   |
| userid     | TEXT  | Foreign key referencing the `username` field in the `Users` table |

**Example Values:**

| sound_name   | file_name             | userid |
| ---------- | --------------------- | ------ |
| Bop        | rick_bop_long.wav     | rick_astley |
| Sample Sound | sample_sound_long.wav | rick_astley |

**Relationship between Tables:**

The `SavedSounds` table has a foreign key `userid` that references the `username` field in the `Users` table. This establishes a relationship between saved sounds and the user who saved them.  This ensures data integrity and allows for efficient retrieval of a user's saved sounds.
