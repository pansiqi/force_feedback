class Status:
    """
    |   状态  |  标志  |
    ---------------------
    |   连接  |   01   |
    |   发送  |   02   |
    |   开始  |   03   |
    |   结束  |   04   |
    |   错误  |   05   |
    """
    connect = 1
    send = 2
    begin = 3
    end = 4
    error = 5


def get_connect_status():
    return Status.connect


def get_send_status():
    return Status.send


def get_begin_status():
    return Status.begin


def get_end_status():
    return Status.end

def get_error_status():
    return Status.error
