from DBSCAN.Dbscan import Dbscan
from Image_analysis.opencv_user.python_call_cc import opencv_call_cc_process
from Pycnn.precast import precast
from Deal_data.tools import tools
filename = "C:\\Users\\谭雯雯\\Desktop\\潘思岐\\img\\test\\3.png"
img = tools.cv_imread(filename)
opencv_call_cc_process(filename)
    # img_show(img)
dbscan = Dbscan(eps=3, min_samples=4)
dbscan.dbsacan_fit(img)
dbscan.data_distribution()
dbscan.get_result()
dbscan.cluster_visual()
print("预估力{}".format(dbscan.calc_mean()))
precast(filename)