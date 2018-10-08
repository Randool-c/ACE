import socket
import json


def sendmessage():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('127.0.0.1', 8080))
    print(s.recv(1024).decode())
    # s.send("["
    #        "{'id': '1', 'links': ['2', '2', '2', '2', '2', '4', '4', '4', '4']},"
    #        "{'id': '2', 'links': ['1', '1', '1', '1', '1', '3', '3', '4', '4', '4', '4', '4', '4', '4', '5', '5', '5']},"
    #        "{'id': '3', 'links': ['2', '2', '4', '4']},"
    #        "{'id': '4', 'links': ['1', '1', '1', '1', '2', '2', '2', '2', '2', '2', '2', '3', '3', '5', '5', '5', '5', '5', '5']},"
    #        "{'id': '5', 'links': ['2', '2', '2', '4', '4', '4', '4', '4', '4']}"
    #        "]".encode(encoding='utf-8'))
    # s.send('exit'.encode(encoding='utf-8'))
    with open('data.json', 'r') as file:
        data = file.read()
        s.send(data.encode(encoding='utf-8'))

    s.close()


sendmessage()
