__author__ = "Yossi"

import asyncio

# from  tcp_by_size import send_with_size ,recv_by_size


SIZE_HEADER_FORMAT = "000000000|"  # n digits for data size + one delimiter
size_header_size = len(SIZE_HEADER_FORMAT)


async def recv_by_size(sock, loop: asyncio.AbstractEventLoop):
    size_header = b""
    data_len = 0
    while len(size_header) < size_header_size:
        _s = await loop.sock_recv(sock, size_header_size - len(size_header))
        if _s == b"":
            size_header = b""
            break
        size_header += _s
    data = b""
    if size_header != b"":
        try:
            data_len = int(size_header[: size_header_size - 1])
            while len(data) < data_len:
                _d = await loop.sock_recv(sock, data_len - len(data))
                if _d == b"":
                    data = b""
                    break
                data += _d
        except:
            return b"message has invalid header"

    if data_len != len(data):
        data = b""  # Partial data is like no data !
    return data


async def send_with_size(sock, bdata, loop: asyncio.AbstractEventLoop):
    if bdata is str:
        bdata = bdata.encode()
    len_data = len(bdata)
    header_data = str(len(bdata)).zfill(size_header_size - 1) + "|"

    bytea = bytearray(header_data, encoding="utf8") + bdata

    await loop.sock_sendall(sock, bytea)
