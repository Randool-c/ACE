import socket
import time


def get_input():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('127.0.0.1', 8080))
    s.listen(1)
    print('waiting for connection...')
    sock, addr = s.accept()
    return tcplink(sock, addr)


def tcplink(sock, addr):
    print('accept new connection from {}'.format(addr))
    sock.send(bytes('Welcome!', encoding='utf-8'))

    data = sock.recv(2 ** 26).decode()
    time.sleep(1)

    sock.send(bytes(data, encoding='utf-8'))
    sock.close()
    print('Connection from {}:{} closed'.format(addr, addr))
    return dict(eval(data))['items']
