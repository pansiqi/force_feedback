from Server.protocol import status
from Server.protocol.protocol import Protocol
from Deal_data.tools import tools

import numpy as np
# TODO:导入数据库包，从而实现数据存入数据库
global dao_data
client_type = 4
filename = ""
def set_global(type):
    global client_type
    client_type = type


def get_global():
    global client_type
    return client_type


class Deal_data:
    def __init__(self, data):
        self.deal_data = data
        self.proto = Protocol()
        self.IsData = True
        self.__connect = 1
        self.__send = 2
        self.__begin = 3
        self.__end = 4
        self.__status = -1  # 默认为-1，表示未连接客户端
        self.__error = 5
        self.data = []
        self.data_length = 0

    def is_data(self):
        if self.deal_data[0] == self.proto.HEAD:
            self.IsData = True
        else:
            self.IsData = False

    def check_data(self):
        s = sum(self.deal_data[2:21])
        if s == self.deal_data[22]:
            return True
        else:
            return status.get_error_status()

    # 如果错误返回错误码 status.get_error_status()

    def get_data_length(self):
        if self.deal_data[1] == 0x00:
            self.data_length = self.deal_data[2]
        else:
            self.data_length = self.deal_data[1] * (16 ** 2) + self.deal_data[2]
        # 或者利用python中的 >> 符号

    def get_client_type(self, data):
        if data == 0x01:
            return 1
        if data == 0x03:
            return 2
        else:
            return 3

    def is_client_status(self):
        var = self.deal_data[4]
        if var == self.proto.CONNECT:
            self.__status = status.get_connect_status()

        elif var == self.proto.BEGIN:
            self.__status = status.get_begin_status()

        elif var == self.proto.HEAD:
            self.__status = status.get_send_status()

        elif var == self.proto.END:
            self.__status = status.get_end_status()

    def get_data(self):
        # 提取有效数据
        self.data = self.deal_data[5:self.data_length - 1]
        print(self.data)
        data2image = self.data
        return tools.wirite_data_to_img(data2image, shape=(2, 7))
        # 加入有效数据处理层

    def get_status(self):
        return self.__status

    def get_error_status(self):
        return self.__error


def data_deal(data):
    deal_data = Deal_data(data)
    deal_data.is_data()
    global client_type
    global filename
    if deal_data.IsData:
        deal_data.is_client_status()
        status = deal_data.get_status()
        dao_data = deal_data.data
        if status == 1:
            print("client connects correctly！")
            print(data[3])
            set_global(deal_data.get_client_type(data[3]))
        elif status == 2:
            print("client prepares to send data! ")
        elif status == 3:
            deal_data.get_data_length()
            if deal_data.check_data() != 5:
                 filename = deal_data.get_data()
            else:
                return status
        elif status == 4:
            print("client finish sending data!")
    else:
        print("HEAD ERROR!")
        return deal_data.get_error_status()


def save_data():
    """

    :return: 存入数据库是否执行成功
    """
    pass


# data = (17, 0, 20, 0, 51, 0, 0, 0, 0, 0, 2, 2, 2, 1, 1, 1, 1, 1, 2, 2, 0, 0, 86, 22)
# data_deal(data)
