from Server.server.force_server import WlwServer
ip = "192.168.31.193"
port = 7000
list = 50  # 最大连接数
from Deal_data.deal_data import filename
from DBSCAN.cluster_opencv import DBSCAN
from Image_analysis.opencv_user import python_call_cc
from Pycnn.precast import precast

def main():
    # 创建一个web服务器
    wlw_server = WlwServer(ip=ip, port=port)
    print("server is running!")
    wlw_server.run_forever()
    filepath = process_img()
    # TODO:整合模块，调用opencv模块，DBSCAN模块，CNN模块

def moduel_call():
    """
    调用各模块
    :return:
    """
    pass

def process_img():
    filenames = []
    if filenames:
        if filenames[-1] == filename:
            pass
        else:
            filenames.append(filename)
        try:
            filepath = filenames.pop(0)
        except:
            print("error")
        finally:
            pass
    else:
        filenames.append(filename)
        print("")
    return filepath




if __name__ == '__main__':
    main()


