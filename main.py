import asyncio
import atexit
import hashlib
import os
import random
import socket
import sys
import wave
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from loguru import logger

import DBHelper
import counter
from send_receive_encrypted import new_client_key, recv_decrypted
from tcp_by_size import send_with_size, recv_by_size

# Thread for shutting down
executor = ThreadPoolExecutor(max_workers=10)

# Every user short file sound
file_short_record = {}

# variables for the long record. package chunk (will probably be removed), And how similar the sound need to be.
FRAMES_PER_SECOND = 48000
SIZE_TO_CHECK = 40000
similarity_threshold = 0.32

# variables for the encryption
p_str = "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A63A3620FFFFFFFFFFFFFFFF"
G = 2
P = int(p_str, 16)
A = random.randint(3, 5000)
srv_DPH_key = int(pow(G, A, P))


async def prepare_hybrid_encryption(sock, client_addr, loop):
    # Get an encryption request from the user
    await recv_by_size(sock, loop)

    # Send the server's Diffie-Hellman public key (along with other parameters)
    await send_with_size(sock, f"{srv_DPH_key}|{G}|{P}".encode(), loop)

    # Receive the client's Diffie-Hellman public key
    cli_dph_key = await recv_by_size(sock, loop)

    # Calculate the shared key
    secret_key = int(pow(int.from_bytes(cli_dph_key, "big"), A, P))

    # Hash the shared key with SHA-256 and take the first 16 bytes
    secret_key = hashlib.sha256(str(secret_key).encode()).digest()[:16]

    # Receive IV from the client
    iv = await recv_by_size(sock, loop)

    logger.info(
        "server DPH key: {}| P: {}\nclient DPH key: {}| shared key: {}\nclient iv: {}",
        srv_DPH_key,
        P,
        cli_dph_key,
        secret_key,
        iv,
    )

    new_client_key(client_addr, secret_key, iv)


def login_user(username, password, DB, login_timeout_task):
    """
    log in user into DB
    and return success or failure type
    """
    if DB.check_username_password(username, password):
        logger.info("User {} connected", username)
        # If client logged in, cancel the login timeout task
        login_timeout_task.cancel()
        return "Username and password match"
    else:
        logger.info("Invalid login attempt for user {}", username)
        return "Username and password do not match"


def sign_up_user(username, password, DB):
    """
    sign up user into DB
    """
    if DB.check_username(username):
        logger.info("Username is in use: {}", username)
        return "Username is in use"
    else:
        if DB.insert_data_to_users_table(username, password):
            logger.info("Signed up successfully: {}, {}", username, password)
            return "Sign up successful"
        else:
            logger.info("Sign up failed: {}, {}", username, password)
            return "Sign up failed"


def save_short_record(username: str, state, content):
    """
    save sound to single use
    """
    logger.info("Got packet: {}, {}", username, state)
    try:
        if username not in file_short_record:  # place all the bytes at the same place
            file_short_record[username] = None
        if file_short_record[username]:
            file_short_record[username] += content
        else:
            file_short_record[username] = content

        if state == "1":  # check if it is the last part
            file_short_record[username] += content
            filename = username + "_short.wav"
            with open(filename, "wb") as file:
                file.write(file_short_record[username])
            file_short_record[username] = None
            logger.info("Saved short record: {}", os.path.abspath(filename))
            return "Saved short record"
        return "Got short record"
    except Exception as e:
        logger.exception("Error saving short record", e)
        return "Error saving record"


def make_short_record(username: str, sound_name, DB):
    """
    convert the saved files into a short record format
    """
    logger.info("Got packet: {}, {}", username, sound_name)
    try:
        exist_file = DB.get_file_name_from_sound(sound_name)
        filename = username + "_short.wav"
        with open(exist_file, "rb") as f_src:  # copy sound into correct format
            with open(filename, "wb") as f_dest:
                # Copy the contents of the original file to the new file
                while True:
                    chunk = f_src.read(1024)
                    if not chunk:
                        break
                    f_dest.write(chunk)
        logger.info("Saved short record: {}", filename)
        return "Saved short record"

    except Exception as e:
        logger.exception("{} Rais general Error", e)
        return "Error making short record"


def count_occurrences(username: str, content: bytes):
    """
    count the number of occurrences of the short sound in the long recording
    """
    logger.info("Got packet: {}", username)
    try:
        filename = username + "_long.wav"

        with wave.open(filename, mode="wb") as wav_file:  # creat sound file
            wav_file.setnchannels(1)
            wav_file.setsampwidth(1)
            wav_file.setframerate(FRAMES_PER_SECOND)
            wav_file.writeframes(content)

        file_size = os.stat(filename).st_size
        logger.info("Wrote {} bytes to {}", file_size, filename)
        sound_file_name = username + str(random.randint(0, 10000)) + "_process_long.wav"
        os.rename(filename, sound_file_name)
        # count occurrences
        logger.info("Sent to process")
        number_of_occurrences = counter.count_similar_sounds(
            username + "_short.wav",
            sound_file_name,
            similarity_threshold,
        )
        logger.info("Number of occurrences: {}", number_of_occurrences)
        #os.remove(sound_file_name)
        return "Number of occurrences: " + str(number_of_occurrences)  #str(4)

    except Exception as e:
        logger.exception("General Error", e)
        return "Error saving record"


def save_record(sound_name: str, username: str, state, content, DB):
    """
    save user sound record for future use
    """
    logger.info("Got packet: {}, {}", username, state)
    try:
        if username not in file_short_record:  # place all the bytes at the same place
            file_short_record[username] = None
        if file_short_record[username]:
            file_short_record[username] += content
        else:
            file_short_record[username] = content

        if state == "1":  # check if it is the last part
            file_short_record[username] += content
            filename = "_" + username + "_" + sound_name + "_saved_Short.wav"
            with open(filename, "wb") as file:  # write into file
                file.write(file_short_record[username])
            file_short_record[username] = None
            DB.insert_data_to_sounds_table(username, sound_name, filename)
            logger.info("Saved file: {}", filename)

            return "Saved file"
        return "Got file part"
    except Exception as e:
        logger.exception("{} Rais general Error", e)
        return "Error saving record"


def return_sound_names(username: str, DB):
    """
    Return user's sound names
    """
    sounds = DB.check_user_sounds(username)
    # turns all the names into one string
    return '~'.join(sounds)


def handle_request(request_code, data, DB, login_timeout_task):
    """
    Handle client request
    string :return: return message to send to client
    """
    try:
        code = request_code.split("~")[0]
        match code:
            case "Login":
                to_send = login_user(
                    request_code.split("~")[1],
                    request_code.split("~")[2],
                    DB,
                    login_timeout_task,
                )
            case "SignUp":
                to_send = sign_up_user(
                    request_code.split("~")[1], request_code.split("~")[2], DB
                )
            case "ShortRecordSave":
                to_send = save_short_record(
                    request_code.split("~")[1], request_code.split("~")[2], data
                )  # name,state,data
            case "ShortRecordExist":
                to_send = make_short_record(
                    request_code.split("~")[1], request_code.split("~")[2], DB
                )  # name,Sound name, database
            case "LongRecord":
                to_send = count_occurrences(
                    request_code.split("~")[1],
                    data,
                )
            case "SaveRecord":
                to_send = save_record(
                    request_code.split("~")[1],
                    request_code.split("~")[2],
                    request_code.split("~")[3],
                    data,
                    DB,
                )  # file name, username,state,data
            case "GetSoundsNames":
                to_send = return_sound_names(request_code.split("~")[1], DB)
            case _:
                logger.exception("unidentified code: {}", request_code)
                to_send = "unidentified code"
        return to_send
    except Exception as err:
        logger.exception("{} Rais general Error", err)
        return "Error: {}", err


async def on_new_client(client_socket: socket, addr):
    """Handles communication with a single client."""
    is_login = False
    db = DBHelper.DBHelper()
    db.create_table()
    loop = asyncio.get_event_loop()
    await prepare_hybrid_encryption(client_socket, addr, loop)
    # Start login timeout task
    login_timeout_task = asyncio.create_task(login_timeout(client_socket))
    while True:
        try:
            if not is_login:
                data = await recv_decrypted(client_socket, addr, loop)
            else:
                data = await recv_by_size(client_socket, loop)
            if not data:
                logger.info("{}  Client disconnected", addr)
                break
            if not data == b'message has invalid header':
                size_to_decode = int(data[0])

                request_code = data[1: size_to_decode + 1].decode("utf-8")
                data = data[size_to_decode + 1:]

                logger.trace("{} >> {}", addr, request_code)
                data = handle_request(request_code, data, db, login_timeout_task)
                if data == "Username and password match":
                    is_login = True
            else:
                logger.info("A message from {} was sent inappropriately", addr)
            await send_with_size(
                client_socket, data.encode("utf-8"), loop
            )  # send data to the client
        except socket.error as e:
            logger.exception("{} Rais Socket Error", addr, e)
            break
        except ValueError as e:
            logger.exception("{} Rais Value Error", addr, e)# kick users that try to use the server without logging in
            # break
        except Exception as e:
            logger.exception("{} Rais general Error", addr, e)

    logger.info("{}  Exited", addr)
    client_socket.close()


async def login_timeout(client_socket):
    try:
        await asyncio.sleep(300)
    except asyncio.CancelledError:
        # Task was cancelled
        return
    # If the client hasn't logged in within 5 minutes, cancel the client socket
    client_socket.close()


@logger.catch
async def main():
    host = "0.0.0.0"
    port = 2525

    with socket.socket(
            socket.AF_INET, socket.SOCK_STREAM
    ) as s:  # Use a context manager to ensure socket closure
        s.bind((host, port))
        s.listen(5)
        s.setblocking(False)

        loop = asyncio.get_event_loop()

        while True:
            c, addr = await loop.sock_accept(s)
            logger.info("New connection from: {}", addr)
            loop.create_task(on_new_client(c, addr))


def on_exit() -> None:
    logger.info("Shutting down")
    executor.shutdown()


if __name__ == "__main__":
    logger.remove()
    logger.add(
        sys.stdout,
        format="<light-blue>{time}</> <lvl>{level: <5}</lvl> [<yellow>{thread.name}</>] [<light-blue>{file}.{"
               "function}:{line}</>] {message}",
    )
    logger.add("debug.log")
    atexit.register(on_exit)
    logger.info("Server ready")
    asyncio.run(main())
