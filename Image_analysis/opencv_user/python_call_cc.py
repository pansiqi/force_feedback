from ctypes import *
import math


#TODO:1.dll编译成release版本，同时优化c++代码
#TODO:2.根据实际数据采集情况，制定opencv图像优化方案，优化参数
#TODO:3.最好能在python中的调用dll的类，这样处理方式由面向过程向面向对象，就不用再申请全局变量，对于python调用来说，方面很多
def opencv_call_cc_process(filename, rho = 1, theta = math.pi/180, minLinLength = 2, maxLineGap = 2, area_scale = 2):
    """

    :return:返回处理图像的数据
    :param filename文件名称
    :param rho 参数极值 以像素值为单位的分辨率. 我们使用 1 像素.
    :param theta 参数极角  以弧度为单位的分辨率. 我们使用 1度 (即CV_PI/180)
    :param threshold: 要”检测” 一条直线所需最少的的曲线交点 
    :param minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.线段的 最小长度
    :param maxLineGap:线段上最近两点之间的阈值
    :param area_scale:力的检测范围
    """
    dll = cdll.LoadLibrary("F:\\code\\force_feedback\\Image_analysis\\opencv_user\\opencv_call_cpp.dll")
    #  动态导入dll文件
    func = dll.opencv_process
    func.argtypes = (POINTER(c_char), c_int, c_double, c_int, c_int, c_int)
    func.restype = c_double
    filename = bytes(filename, "GBK")
    rho = c_int(rho)
    theta = c_double(theta)
    minLinLength = c_int(minLinLength)
    maxLineGap = c_int(maxLineGap)
    angle_mean = func(filename, rho, theta, minLinLength, maxLineGap, area_scale)
    area = c_double.in_dll(dll, "area").value
    postion_x = c_long.in_dll(dll, "postion_x").value
    postion_y = c_long.in_dll(dll, "postion_y").value
    center = [postion_x, postion_y]
    print(center)
    #访问dll全局变量 获取面积范围和中心点
    print(area)
    print("the mean angle is:{}".format(angle_mean))

# opencv_call_cc_process("..//force_feedback//DATA//img//test//1.png")


