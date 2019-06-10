import socket
import threading
import time
from queue import Queue
from struct import *

from Server.conf import config
from Deal_data.deal_data import *

class WlwServer:
    queue = Queue()

    def __init__(self, port, ip):
        """初始化对象"""
        # 创建套接字
        self.tcp_server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # 解决程序端口占用问题
        self.tcp_server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # 绑定本地ip地址
        self.tcp_server_socket.bind((ip, port))
        self.tcp_server_socket.listen(config.get_max_listen_size_value())
        self.sockets = {}  # 创建字典，用来存储socket的信息

    def run_forever(self):
        """设备连接"""
        while True:
            localtime = time.asctime(time.localtime(time.time()))
            #   等待设备连接(通过ip地址和端口建立tcp连接)
            #   如果有设备连接，则会生成用于设备和服务器通讯的套接字：new_socket
            #   会获取到设备的ip地址和端口
            new_socket, addr = self.tcp_server_socket.accept()
            self.sockets[addr[0]] = new_socket
            new_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, config.get_send_buf_size_value())
            print("time:{}".format(localtime) + " client is connected! addr:{}".format(addr))

            #  创建线程处理设备的需求
            t1 = threading.Thread(target=self.service_machine, args=(new_socket, addr))
            t1.start()

    def service_machine(self, new_socket, client_addr):
        """业务处理"""
        while True:
            receive_data = new_socket.recv(config.get_recv_buf_size_value())
            # 4.如果设备发送的数据不为空
            if receive_data:
                # 4.1 打印接收的数据，这里可以将设备发送的数据写入到文件中
                # 获取设备的ID信息
                print(receive_data)
                config.set_temp_data(receive_data)
                self.queue.put(config.get_temp_data(), block=False)
                # 非阻塞入队列
                """
                远程客户端关闭连接，那么对象就会销毁，队列就会消失
                """
                # TODO: 将队列放在主方法里，在主方法里处理数据
                if self.queue != self.queue.empty():
                    data = unpack('24B', self.queue.get())
                    print(data)
                    if data_deal(data) == 5:
                        new_socket.send(bytes('resend！！'))
                    addr = '192.168.31.127'
                    if addr in self.sockets:
                        trans_socket = self.sockets[addr]
                        time.sleep(1)
                        trans_socket.send(receive_data)
                    else:
                        print('设备未上线')
            else:
                print('设备{0}断开连接...'.format(client_addr))
                break
        # 关闭套接字
        new_socket.close()
