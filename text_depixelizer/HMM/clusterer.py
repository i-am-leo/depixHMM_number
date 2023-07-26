from abc import ABC, abstractmethod
from typing import List

import numpy as np
from sklearn.cluster import KMeans

from text_depixelizer.training_pipeline.windows import Window

import matplotlib.pyplot as plt
class Clusterer(ABC):
    centroids: List[np.ndarray]

    @abstractmethod
    def map_windows_to_cluster(self, windows: List[Window]) -> List[Window]:
        pass

    @abstractmethod
    def map_values_to_cluster(self, values: List[np.array]) -> List[int]:
        pass

class KmeansClusterer(Clusterer):
    kmeans: KMeans

    def __init__(self, windows: List[Window], k: int):
        # print([window.values for window in windows])  # 聚类 标量
        # print(len(windows))
        X = np.array([window.values for window in windows])
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(X)
        # 画图表示聚类效果
        # y_pred = KMeans(n_clusters=250, random_state=250).fit_predict(X)
        # plt.scatter(X[:, 0], X[:, 1], c=y_pred)
        # plt.show()
        self.kmeans = kmeans

    def map_windows_to_cluster(self, windows: List[Window]) -> List[Window]:
        k_values: List[int] = self.map_values_to_cluster([window.values for window in windows])
        # print("k_values",k_values)
        for window, k_value in zip(windows, k_values):
            window.k = k_value  # 对应地设置每个窗口的 k 值为对应的簇标签
            # print(window)  # 检查窗口，给每个窗口打标签
        # print(windows[0])
        return windows

    def map_values_to_cluster(self, values: List[np.array]) -> List[int]:
        # print("len(values)", len(values))
        # print("values",values)
        k_values: List[int] = self.kmeans.predict(np.array(values))
        return k_values
