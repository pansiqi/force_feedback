class Protocol():
    def __init__(self):
        self.__HEAD = 0x11  # 帧头
        self.__TAIL = 0x16  # 帧尾
        self.__HEART = 0x34  # 心跳包
        self.__BEGIN = 0x33  # 开始发送
        self.__CONNECT = 0x32  # 连接
        self.__END = 0x35  # 结束

    @property
    def HEAD(self):
        return self.__HEAD

    @property
    def TAIL(self):
        return self.__TAIL

    @property
    def HEART(self):
        return self.__HEART

    @property
    def BEGIN(self):
        return self.__BEGIN

    @property
    def CONNECT(self):
        return self.__CONNECT

    @property
    def END(self):
        return self.__END
