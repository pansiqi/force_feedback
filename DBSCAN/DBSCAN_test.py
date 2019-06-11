from DBSCAN.Dbscan import Dbscan
from Deal_data.tools.tools import cv_imread

# filename = "..//DATA//img//棍//2.png"
filename = "C:\\Users\\谭雯雯\\Desktop\\潘思岐\\img\\test\\3.png"
if __name__ == "__main__":
    img = cv_imread(filename)
    # img_show(img)
    dbscan = Dbscan(eps=3, min_samples=4)
    dbscan.dbsacan_fit(img)
    dbscan.data_distribution()
    dbscan.get_result()
    dbscan.cluster_visual()
    print("预估力{}".format(dbscan.calc_mean()))
