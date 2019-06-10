import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class Dbscan:
    def __init__(self, eps=3, min_samples=4):
        self.data = []
        self.X = np.array([])
        self.labels = []
        self.eps = eps
        self.min_samples = min_samples
        self.n_clusters_ = 0
        self.lock_visiual = False
        self.means = []
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def dbsacan_fit(self, data):
        x = []
        for i in range(16):
            for j in range(16):
                if (data[i][j] != 0):
                    x.append([i, j, data[i][j]])
        self.X = np.array(x)
        self.model.fit(self.X)
        self.labels = self.model.labels_
        self.lock_visiual = True
        return self


    def get_result(self):
        try:
            self.n_clusters_ = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        except ValueError:
            print("未进行聚类或者聚类失败")
        finally:
            pass
        return self.model.labels_, self.n_clusters_


    def data_distribution(self):
        for k in range(len(self.X)):
          plt.scatter(self.X[k][1], 16-self.X[k][0])
        plt.show()

    def cluster_visual(self):
        '''
        聚类可视化
        :return:
        '''
        if(self.lock_visiual):
            if(self.n_clusters_ != 0):
                core_samples_mask = np.zeros_like(self.labels, dtype=bool)
                core_samples_mask[self.model.core_sample_indices_] = True
                # unique_labels = set(self.labels)
                # means = []
                marker = ['v', '^', 'o', 'x', '+']
                cols = ["r", "g", "b", "c", "m", "y"]
                for i in range(self.n_clusters_):
                    plt.scatter(self.X[self.labels == i][:, 1], 16 - self.X[self.labels == i][:, 0], c=cols[i], marker=marker[i],
                                label="label" + str(i))
                    mean = np.sum(self.X[self.labels == i][:, 2]) / len((self.X[self.labels == i][:, 2]))
                    self.means.append(mean)
                plt.scatter(self.X[self.labels == -1][:, 1], 16 - self.X[self.labels == -1][:, 0], c="k", marker="*",
                        label="noise")
                plt.xlabel('sepal length')
                plt.ylabel('sepal width')
                plt.legend(loc=2)
                plt.show()
            else:
                print("未获得聚类种数！！！")
        else:
            print("请先进行聚类")


        return

    def calc_mean(self):
        if self.means:
            max = self.means[0]
            for i in range(len(self.means)):
                if self.means[i] > max:
                    max = self.means[i]
                else:
                    max = self.means[0]
            return max
        else:
            print("聚类数为空！！！")

