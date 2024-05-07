from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from base64 import b64encode, b64decode
import tcp_by_size

CLIENTS_AES_INFO = {}


def new_client_key(client_addr, aes_key, iv):
    CLIENTS_AES_INFO[client_addr] = (aes_key, iv)
    print(CLIENTS_AES_INFO)


def send_encrypted(sock, client_addr, to_send, loop):
    encrypt_cipher = AES.new(
        CLIENTS_AES_INFO[client_addr][0], AES.MODE_CBC, CLIENTS_AES_INFO[client_addr][1]
    )
    to_send = encrypt_cipher.encrypt(pad(to_send, AES.block_size))
    tcp_by_size.send_with_size(sock, b64encode(to_send), loop)


async def recv_decrypted(sock, client_addr, loop):
    decrypt_cipher = AES.new(
        CLIENTS_AES_INFO[client_addr][0], AES.MODE_CBC, CLIENTS_AES_INFO[client_addr][1]
    )
    received = await tcp_by_size.recv_by_size(sock, loop)
    if received == b"":
        return b""
    original_data = unpad(
        decrypt_cipher.decrypt(b64decode(received)), AES.block_size
    )  # .decode().strip())  # Decrypt and then up-pad the result
    return original_data
