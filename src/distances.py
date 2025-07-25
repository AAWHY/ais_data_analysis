from haversine import haversine
import numpy as np
from fastdtw import fastdtw
import pickle
import os
from joblib import Parallel, delayed
from itertools import product
import time
from numba import jit
from src.frechet_dist import fast_frechet
import hausdorff


def dict_reorder(x):
    """
    对字典按key排序（递归）
    :param x: dict格式数据
    :return: 排序后的dict
    """
    return {k: dict_reorder(v) if isinstance(v, dict) else v for k, v in sorted(x.items())}


@jit(forceobj=True)
def MD(a, b):
    """
    GPS轨迹的Merge Distance距离计算
    :param a: 轨迹A
    :param b: 轨迹B
    :return: merge距离

    参考文献：
    [1] Ismail, Anas, and Antoine Vigneron. "A new trajectory similarity measure for GPS data." Proceedings of the 6th ACM SIGSPATIAL International Workshop on GeoStreaming. 2015.
    [2] Li, Huanhuan, et al. "Spatio-temporal vessel trajectory clustering based on data mapping and density." IEEE Access 6 (2018): 58939-58954.
    """
    m = len(a)
    n = len(b)
    A = np.zeros([m, n])
    B = np.zeros([m, n])

    # 计算轨迹a和b的分段距离
    a_dist = [haversine(a[i-1], a[i]) for i in range(1, m)]
    b_dist = [haversine(b[i-1], b[i]) for i in range(1, n)]

    # 初始化边界
    i = 0
    a_d = 0
    for j in range(n):
        k = j - 1
        if k > 0 and k < n:
            a_d = a_d + b_dist[k-1]
        A[i, j] = a_d + haversine(b[j], a[0])

    j = 0
    b_d = 0
    for i in range(m):
        k = i - 1
        if k > 0 and k < n:
            b_d = b_d + a_dist[k - 1]
        B[i, j] = b_d + haversine(a[i], b[0])

    # 计算A和B的第一行/列
    j = 0
    for i in range(1, m):
        A[i, j] = min(A[i - 1, j] + a_dist[i - 1], B[i - 1, j] + haversine(a[i], b[j]))
    i = 0
    for j in range(1, n):
        B[i, j] = min(A[i, j - 1] + haversine(b[j], a[i]), B[i, j - 1] + b_dist[j - 1])

    # 动态规划填充A和B
    for i, j in product(range(1, m), range(1, n)):
        A[i, j] = min(A[i-1, j] + a_dist[i-1], B[i-1, j] + haversine(a[i], b[j]))
        B[i, j] = min(A[i, j-1] + haversine(b[j], a[i]), B[i, j-1] + b_dist[j-1])

    # 获取merge距离
    md_dist = min(A[-1, -1], B[-1, -1])
    if md_dist != 0:
        md_dist = ((2*md_dist) / (np.sum(a_dist)+np.sum(b_dist)))-1
    return md_dist


### 并行计算距离的辅助函数 ###
# @jit(forceobj=True)
def _dist_func(dataset, metric, mmsis, dim_set, id_b, id_a, s_a, dist_matrix, process_time):
    """
    计算两条轨迹之间的距离（支持多种距离度量）
    :param dataset: 轨迹数据集（dict）
    :param metric: 距离度量方法
    :param mmsis: 所有轨迹的mmsi列表
    :param dim_set: 维度名（lat/lon）
    :param id_b: 轨迹b的索引
    :param id_a: 轨迹a的索引
    :param s_a: 轨迹a的坐标
    :param dist_matrix: 距离矩阵
    :param process_time: 处理时间矩阵
    :return: 距离矩阵和处理时间
    """
    # 获取轨迹b
    t0 = time.time()
    s_b = [dataset[mmsis[id_b]][dim] for dim in dim_set]
    # 选择距离度量方式
    if metric == 'dtw':
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = fastdtw(np.array(s_a).T, np.array(s_b).T, dist=haversine)[0]
    elif metric == 'hd':
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = hausdorff.hausdorff_distance(np.array(s_a).T, np.array(s_b).T, 'haversine')
    elif metric == 'dfd':
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = fast_frechet(np.array(s_a).T, np.array(s_b).T)
    else:
        dist_matrix[mmsis[id_a]][mmsis[id_b]] = MD(np.array(s_a).T, np.array(s_b).T)
    print(f'dist = {id_a}, {id_b} = {dist_matrix[mmsis[id_a]][mmsis[id_b]]}')
    dist_matrix[mmsis[id_b]][mmsis[id_a]] = dist_matrix[mmsis[id_a]][mmsis[id_b]]
    t1 = time.time() - t0
    process_time[mmsis[id_a]][mmsis[id_b]] = t1
    process_time[mmsis[id_b]][mmsis[id_a]] = t1


def compute_distance_matrix(dataset, path, verbose=True, njobs=15, metric='dtw'):
    """
    计算整个数据集的距离矩阵，并保存到文件
    :param dataset: 轨迹数据集（dict）
    :param path: 结果保存路径
    :param verbose: 是否打印进度信息
    :param njobs: 并行核数
    :param metric: 距离度量方法
    :return: 距离矩阵文件路径和耗时文件路径
    """
    if not os.path.exists(f'{path}/distances.p'):
        _dim_set = ['lat', 'lon']
        _mmsis = list(dataset.keys())

        dist_matrix = {}
        process_time = {}
        for id_a in range(len(_mmsis)):
            dist_matrix[_mmsis[id_a]] = {}
            process_time[_mmsis[id_a]] = {}

        for id_a in range(len(_mmsis)):
            if verbose:
                print(f"{metric}: {id_a} of {len(_mmsis)}")
            dist_matrix[_mmsis[id_a]][_mmsis[id_a]] = 0
            # 获取轨迹a
            s_a = [dataset[_mmsis[id_a]][dim] for dim in _dim_set]
            # 并行计算与其它轨迹的距离
            Parallel(n_jobs=njobs, require='sharedmem')(delayed(_dist_func)(dataset, metric, _mmsis, _dim_set, id_b, id_a,
                                                                            s_a, dist_matrix, process_time)
                                                        for id_b in list(range(id_a + 1, len(_mmsis))))
        # 排序并保存结果
        dist_matrix = dict_reorder(dist_matrix)
        process_time = dict_reorder(process_time)
        dm = np.array([list(item.values()) for item in dist_matrix.values()])

        # 保存距离矩阵和耗时
        os.makedirs(path, exist_ok=True)
        pickle.dump(dm, open(f'{path}/distances.p', 'wb'))
        pickle.dump(process_time, open(f'{path}/distances_time.p', 'wb'))
    else:
        if verbose:
            print('\tDistances already computed.')

    # 返回保存的文件路径
    dm_path = f'{path}/distances.p'
    process_time_path = f'{path}/distances_time.p'

    return dm_path, process_time_path
