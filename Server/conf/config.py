class Global_var:
    TEMP_DATA = []  # 设置跨文件数据共享临时变量
    RECV_BUF_SIZE = 3000
    SEND_BUF_SIZE = 3000
    MAX_LISTEN_SIZE = 100


def set_temp_data(data):
    Global_var.TEMP_DATA = data


def get_temp_data():
    return Global_var.TEMP_DATA


def get_recv_buf_size_value():
    return Global_var.RECV_BUF_SIZE


def set_recv_buf_size_value(value):
    Global_var.RECV_BUF_SIZE = value


def get_send_buf_size_value():
    return Global_var.SEND_BUF_SIZE


def set_send_buf_size_value(value):
    Global_var.SEND_BUF_SIZE = value


def set_max_listen_size_value(value):
    Global_var.MAX_LISTEN_SIZE = value


def get_max_listen_size_value():
    return Global_var.MAX_LISTEN_SIZE
