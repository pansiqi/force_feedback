import time

import cv2
import numpy as np

def cv_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), 1)
    if(cv_img.ndim == 3):
        print("image's dimension is {}".format(cv_img.ndim))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
    return cv_img

def img_show(img):
    height, width = img.shape[:2]
    imgout = cv2.resize(img, dsize=(height * 32, width * 32), interpolation=cv2.INTER_AREA)
    cv2.imshow("display", imgout)
    cv2.waitKey(0)


def wirite_data_to_img(data, shape=()):
    data = np.array(data)
    data.reshape(shape)
    filename = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()) + "_input.png"
    cv2.imwrite("..\\..\\DATA\\input_images\\"+filename, data)
    print(filename)
    return filename

def test_data_write():
    datas = []
    data = (0, 0, 0, 0, 0, 2, 255, 255, 1, 1, 1, 1, 1, 2, 255, 255)
    # 将数据变成list类型
    data = list(data)
    for i in range(16):
        datas.append(data)
    # np.array(datas)
    # np.reshape(datas, (16, 16))
    datas = np.array(datas)
    print(datas.shape)
    print(datas)
    return datas
# wirite_data_to_img(test_data_write(),shape=(16,16))
# path = "..//img//test//3.png"
# test_file_path = "..//img//1.jpeg"
# # RGB---->灰度图
# # 公式 Y' = 0.299 R + 0.587 G + 0.114 B
# cv_img = cv2.imdecode(np.fromfile(test_file_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
# print(cv_img.dtype)
# print(cv_img)
