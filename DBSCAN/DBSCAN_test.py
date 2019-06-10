from Deal_data.tools.tools import cv_imread
from Deal_data.tools.tools import img_show

from DBSCAN.Dbscan import Dbscan


filename = "..//DATA//img//棍//2.png"

if __name__  == "__main__":
    img = cv_imread(filename)
    img_show(img)
    dbscan = Dbscan(eps=3, min_samples=4)
    dbscan.dbsacan_fit(img)
    dbscan.data_distribution()
    dbscan.get_result()
    print("预估力{}".format(dbscan.calc_mean()))
