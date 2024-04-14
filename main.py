import atexit
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib import request
import os

from loguru import logger

from tcp_by_size import send_with_size, recv_by_size
import DBHelper
import counter

executor = ThreadPoolExecutor(max_workers=10)
file_short_record = {}

SIZE_TO_CHECK = 20000
similarity_threshold = 0.7


def login_user(username, password, DB):
    if DB.check_username_password(username, password):
        logger.info("Username and password match: {}, {}", username, password)
        return "Username and password match"
    else:
        logger.info("Username and password do not match: {}, {}", username, password)
        return "Username and password do not match"


def sign_up_user(username, password, DB):
    if DB.check_username(username):
        logger.info("Username is in use: {}", username)
        return "Username is in use"
    else:
        if DB.insert_data(username, password):
            logger.info("Signed up successfully: {}, {}", username, password)
            return "Sign up successful"
        else:
            logger.info("Sign up failed: {}, {}", username, password)
            return "Sign up failed"


def save_short_record(username: str, state, content):
    logger.info("Got packet: {}, {}", username, state)
    try:
        if username not in file_short_record:
            file_short_record[username] = None
        if file_short_record[username]:
            file_short_record[username] += content
        else:
            file_short_record[username] = content

        if state == "1":
            file_short_record[username] += content
            filename = username + "_short.ogg"
            with open(filename, "wb") as file:
                file.write(file_short_record[username])
            file_short_record[username] = None
            logger.info("Saved short record: {}", filename)
            return "Saved short record"
        return "Got short record"
    except Exception as e:
        logger.exception("{} Rais general Error", e)
        return "Error saving record"


def count_occurrences(username: str, state, readSize, content):
    logger.info("Got packet: {}, {}", username, state)
    try:
        filename = username + "_long.ogg"
        with open(filename, "ab") as file:
            file.write(content)

        file_size = os.stat(filename).st_size

        if state == "1" or readSize < 1024 or file_size >= SIZE_TO_CHECK:
            sound_file_name = username + "_process_long.ogg"
            os.rename(filename, sound_file_name)
            logger.info("Sent to process")
            number_of_occurrences = counter.count_similar_sounds(
                file_short_record[username],
                sound_file_name,
                similarity_threshold,
            )
            logger.info("Number of occurrences: {}", number_of_occurrences)
            os.remove(sound_file_name)
            return number_of_occurrences

        return "Got short record"
    except Exception as e:
        logger.exception("{} Rais general Error", e)
        return "Error saving record"


def save_record(data):
    # Your logic for saving record here
    pass


def handle_request(request_code, data, DB):
    """
    Handle client request
    string :return: return message to send to client
    """
    to_send = " "
    try:
        code = request_code.split("~")[0]
        match code:
            case "Login":
                to_send = login_user(
                    request_code.split("~")[1], request_code.split("~")[2], DB
                )
            case "SignUp":
                to_send = sign_up_user(
                    request_code.split("~")[1], request_code.split("~")[2], DB
                )
            case "ShortRecord":
                to_send = save_short_record(
                    request_code.split("~")[1], request_code.split("~")[2], data
                )  # name,state,data
            case "LongRecord":
                to_send = count_occurrences(data.split("~")[1])
            case "SaveRecord":
                to_send = save_record(data.split("~")[1])
            case _:
                logger.info("unidentified code: {}", request_code)
                to_send = "unidentified code"
        return to_send
    except Exception as err:
        logger.exception("{} Rais general Error", err)
        return "Error: {}", err


# ShortRecoed~0~00011101010101
# ShortRecoed~1~00011101010101
def on_new_client(client_socket: socket, addr):
    """Handles communication with a single client."""
    DB = DBHelper.DBHelper()
    while True:
        try:
            data = recv_by_size(client_socket)
            if not data:
                logger.info("{}  Client disconnected", addr)
                break
            size_to_decode = int(data[0])
            request_code = data[1 : size_to_decode + 1].decode("utf-8")
            data = data[size_to_decode + 1 :]

            logger.info("{} >> {}", addr, request_code)
            data = handle_request(request_code, data, DB)

            send_with_size(
                client_socket, data.encode("utf-8")
            )  # send data to the client
        except socket.error as e:
            logger.exception("{} Rais Socket Error", addr, e)
            break
        except Exception as e:
            logger.exception("{} Rais general Error", addr, e)

    logger.info("{}  Exited", addr)
    client_socket.close()


@logger.catch
def main():
    host = "0.0.0.0"
    port = 2525

    with socket.socket() as s:  # Use a context manager to ensure socket closure
        s.bind((host, port))
        s.listen(5)

        while True:
            c, addr = s.accept()
            logger.info("New connection from: {}", addr)
            executor.submit(on_new_client, c, addr)


def on_exit() -> None:
    logger.info("Shutting down")
    executor.shutdown()


if __name__ == "__main__":
    atexit.register(on_exit)
    logger.info("Server ready")
    main()
