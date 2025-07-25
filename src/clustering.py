import os, pickle
import numpy as np
import pandas as pd
import time
import hdbscan
import matplotlib.pyplot as plt


class Clustering:
    def __init__(self, ais_data_path, distance_matrix_path, folder, minClusterSize=2, verbose=True, **args):
        """
        初始化聚类对象，加载距离矩阵并进行归一化处理，随后执行聚类。
        :param ais_data_path: AIS数据集路径
        :param distance_matrix_path: 轨迹距离矩阵路径
        :param folder: 结果保存文件夹
        :param minClusterSize: HDBSCAN最小聚类数
        :param verbose: 是否打印信息
        """
        self.ais_data_path = ais_data_path
        self._verbose = verbose
        # 加载距离矩阵
        self.dm = abs(pickle.load(open(distance_matrix_path, 'rb')))
        self.minClusterSize = minClusterSize
        self._model = None
        self.labels = None

        # 处理无穷大距离
        self.dm[np.isinf(self.dm)] = self.dm[~np.isinf(self.dm)].max() + 1
        # 距离归一化
        if 'norm_dist' in args.keys():
            if args['norm_dist']:
                if (self.dm < 0).sum() > 0:
                    self.dm = abs(self.dm)
                self.dm = self.dm/self.dm.max()

        # 结果保存路径
        self.path = f'{folder}/hdbscan'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        self.results_file_path = f'{self.path}/hdbscan.csv'
        self.labels_file_path = f'{self.path}/labels_hdbscan.csv'
        self.time_path = f'{self.path}/time_hdbscan.csv'

        # 执行聚类并计时
        t0 = time.time()
        self.computer_clustering()
        t1 = time.time() - t0
        self.time_elapsed = t1
        pickle.dump(self.time_elapsed, open(self.time_path, 'wb'))

    def computer_clustering(self):
        """
        执行HDBSCAN聚类算法，并保存聚类树图像。
        """
        if self._verbose:
            print(f'Clustering data using HDBSCAN')

        # 使用预计算距离矩阵进行聚类
        self._model = hdbscan.HDBSCAN(min_cluster_size=self.minClusterSize, min_samples=1, allow_single_cluster=True, metric='precomputed')
        self._model.fit(self.dm)
        # 绘制并保存聚类树（dendrogram）
        axis = self._model.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
        # plt.show()
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.savefig(f'{self.path}/dendogram-{self.minClusterSize}.png', bbox_inches='tight')
        plt.close()
        # 获取聚类标签
        self.labels = self._model.labels_
        self._agg_cluster_labels()

    def _agg_cluster_labels(self):
        """
        将聚类标签信息加入原始数据，并保存聚类结果和标签。
        """
        data = pd.read_csv(self.ais_data_path)
        # 兼容不同的轨迹列名
        if not 'trips' in data.columns:
            data = data.rename(columns={'trajectory': 'trips'})
        # 构建标签字典
        labels = pd.DataFrame([self.labels], columns=data['trips'].unique()).to_dict('records')[0]
        aux = data['trips']
        aux = aux.map(labels)
        aux.name = 'Clusters'
        # 合并聚类标签
        cluster_dataset = pd.concat([data, aux], axis=1)
        labels_mmsi = cluster_dataset[['mmsi', 'trips', 'Clusters']].drop_duplicates()
        # 保存完整聚类结果和标签
        cluster_dataset.to_csv(self.results_file_path)
        labels_mmsi.to_csv(self.labels_file_path)

